backbone
-------------
0. Export the env HDPL_TOOLCHAIN_ITVM_INSTALL to the path where tvm is installed.

1. Generate the module params json
  python3 store_backbone.py

2. Run the network
  cmake .
  make
  ./hdpl_backbone_run

3. remove the Intermediate file
  rm -rf CMakeFiles libbackbone.* Makefile CMakeCache.txt  cmake_install.cmake hdpl_backbone_run *.bin

