pointpillars
-------------
0. Export the env HDPL_TOOLCHAIN_ITVM_INSTALL to the path where tvm is installed.

1. Generate the module params json
  python3 store_pointpillars.py

2. Run the network
  mkdir build
  cd build
  cmake ..
  make
  cd ..
  ./build/hdpl_pointpillars_run

3. remove the Intermediate file
  rm -rf build libpointpillars2.* pointpillars_rpn.zip pointpillars_rpn

