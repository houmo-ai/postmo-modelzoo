#!/usr/bin/env bash
set -e

SCRIPT_PATH=$(cd "$(dirname ${BASH_SOURCE[0]})"; pwd)

cpp_test() {
  store="store_${1}.py"
  run="./hdpl_${1}_run"
  pushd ${SCRIPT_PATH}/${1}
    echo "Testing python3 ${store} ..."
    python3 $store
    if [ $? != 0 ]
    then
      echo "Error: homo ${store} run faild."
      exit 1
    fi
    cmake .
    make
    $run
    if [ $? != 0 ]
    then
      echo "Error: homo ${run} faild."
      exit 1
    fi
  popd
}

cpp_test 'backbone'
cpp_test 'box_head'
cpp_test 'attr_head'
cpp_test 'backbone_dlopen'
cpp_test 'light_recog'
cpp_test 'pointpillars'