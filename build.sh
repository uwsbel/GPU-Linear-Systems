#!/bin/bash

# Common compiler flags
CXXFLAGS="-Wall -O3 -std=c++17 -fopenmp"

# Build the naive implementation
g++ $CXXFLAGS task1.cpp implementations/lu_naive.cpp -o task1_naive

# Build the MKL implementation
g++ $CXXFLAGS task1.cpp implementations/lu_mkl.cpp -o task1_mkl \
-I/opt/intel/oneapi/mkl/latest/include \
-L/opt/intel/oneapi/mkl/latest/lib/intel64 \
-lmkl_rt

