boxhead
-------------
0. Export the env HDPL_TOOLCHAIN_ITVM_INSTALL to the path where tvm is installed.

1. Generate the module params json with softmax
  python3 store_box_head.py

2.  Generate the module params json without softmax
  python3 store_box_head_without_softmax.py

3. Run the network
  cmake .
  make
  ./hdpl_box_head_run

4. remove the Intermediate file
  rm -rf CMakeFiles libboxhead.* Makefile CMakeCache.txt  cmake_install.cmake hdpl_box_head_run *.bin

