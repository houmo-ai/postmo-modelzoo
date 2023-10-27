cruise_cutin_vehicle
-------------
0. Export the env HDPL_TOOLCHAIN_ITVM_INSTALL to the path where tvm is installed.

1. Generate the module params json with softmax
  python3 store_cruise_cutin_vehicle.py

2. Run the network
  cmake .
  make
  ./hdpl_cruise_cutin_vehicle_run

3. remove the Intermediate file
  rm -rf CMakeFiles libcruise_cutin_vehicle.* Makefile CMakeCache.txt  cmake_install.cmake hdpl_cruise_cutin_vehicle_run *.bin tensorlist.json cruise_cutin_vehicle.zip cruise_cutin_vehicle

