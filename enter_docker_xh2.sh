#! /bin/bash

# 0. 必要变量
VERSION=develop
IMAGE_NAME="harbor.houmo.ai/toolchain/release:Dadao-xh2-${VERSION}-ubuntu20.04-x86.64.latest"
CONTAINER_NAME="$(whoami).Dadao_xh2_${VERSION}"
CONTAINER_HOME="/container/$(whoami)"
USER_CONFIG="-v /develop02:/develop02 -v /data:/data"

PRINT_RED() { echo -e "\033[1;31m$@\033[0m"; }
PRINT_GREEN() { echo -e "\033[1;32m$@\033[0m"; }
PRINT_BLUE() { echo -e "\033[1;34m$@\033[0m"; }
RUN_IN_DOCKER() { docker exec -it $CONTAINER_NAME bash -c "$@"; }
ENTER_DOCKER() {
  PRINT_BLUE "Re-enter the container using:";
  PRINT_BLUE "-> docker exec -it ${CONTAINER_NAME} bash";
  PRINT_GREEN "Enter the container now.";
  docker exec -it ${CONTAINER_NAME} bash;
}

PRINT_BLUE "start container named \"$CONTAINER_NAME\" with image $IMAGE_NAME"

# 1. 挂载路径设置，容器内路径相同
VOLUME_HOME="/Dadao_xh2_${VERSION}"

# 2. [非必要]处理命令行参数，可选参数只有"restart"
if [ $# -gt 0 ]; then
  if [ "$1" == "restart" ]; then
    PRINT_BLUE "docker stop $CONTAINER_NAME"
    docker stop $CONTAINER_NAME >/dev/null
    docker rm $CONTAINER_NAME >/dev/null
  else
    PRINT_RED "unknown argument"; exit
  fi
else
  if docker ps -a | grep -q $CONTAINER_NAME; then
    PRINT_BLUE "container exists"; 
    ENTER_DOCKER
    exit
  else
    PRINT_BLUE "creating a new container"
  fi
fi

# 3. 以后台状态创建容器，并挂载第1步设置的路径
PRINT_BLUE "docker pull $IMAGE_NAME";
docker pull $IMAGE_NAME >/dev/null
docker run --privileged --network=host --pid=host \
  -v $(pwd):$VOLUME_HOME -v $HOME:$HOME $USER_CONFIG \
  --name $CONTAINER_NAME -w $VOLUME_HOME -itd -u $(id -u):$(id -g) $IMAGE_NAME >/dev/null

# 4. 创建与宿主相同的用户，并赋予sudo权限
docker exec -u 0:0 -it $CONTAINER_NAME bash -c "mkdir -p /container && chown $(id -u):$(id -g) /container"
docker exec -u 0:0 -it $CONTAINER_NAME bash -c "groupadd -g $(id -g) $(whoami) && useradd -m -d ${CONTAINER_HOME} -u $(id -u) -g $(whoami) $(whoami) && usermod -a -G sudo $(whoami) && echo \"$(whoami) ALL=(ALL) NOPASSWD: ALL\">> /etc/sudoers"

# 5. 进入docker容器
ENTER_DOCKER
