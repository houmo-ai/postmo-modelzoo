#!/bin/bash
if [ $(uname -m) = "x86_64" ] && [ "$HOUMO_TARGET" = "xh2" ]; then
  cd $PWD/../3rdparty/eigen-3.4.0
  if [ -e build ]; then
    rm -rf build
  fi
  mkdir build && cd build
  cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
  sudo make install
else
  if [ $(uname -m) != "x86_64" ]; then
    echo "UnSupport Platform" $(uname -m)
  fi

  if [ "$HOUMO_TARGET" = "xh2" ]; then
    echo "No xh2Backhend found!"
  fi
fi
