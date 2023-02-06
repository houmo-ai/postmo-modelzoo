backbone_dlopen
-------------
0. Export the env HDPL_TOOLCHAIN_ITVM_INSTALL to the path where tvm is installed.

1. Generate the module params json
  python3 store_backbone_dlopen.py

2. Run the network
  cmake .
  make
  ./hdpl_backbone_dlopen_run

3. remove the Intermediate file
  rm -rf CMakeFiles libbackbone.* Makefile CMakeCache.txt  cmake_install.cmake hdpl_backbone_dlopen_run *.bin *.zip libhdpl_backbone_run.so backbone

