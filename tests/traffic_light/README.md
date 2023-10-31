light_recog
-------------
0. Export the env HDPL_TOOLCHAIN_ITVM_INSTALL to the path where tvm is installed.

1. Generate the module params json with softmax
  python3 store_light_recog.py

2. Run the network
  cmake .
  make
  ./hdpl_light_recog_run

3. remove the Intermediate file
  rm -rf CMakeFiles liblight_recog.* Makefile CMakeCache.txt  cmake_install.cmake hdpl_light_recog_run *.bin traffic_light_recog.zip traffic_light_recog
