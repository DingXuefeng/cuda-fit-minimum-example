#!/bin/bash

mkdir .build
cd .build
cmake .. -DCMAKE_INSTALL_PREFIX=..
make -j install
