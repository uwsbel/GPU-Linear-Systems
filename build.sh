#!/bin/bash

# Common compiler flags
CXXFLAGS="-Wall -O3 -std=c++17 -fopenmp -DEIGEN_USE_MKL_ALL -DEIGEN_USE_LAPACKE"

# Build the naive implementation
g++ $CXXFLAGS task1.cpp implementations/lu_naive.cpp -o task1_naive

# Build the MKL implementation
g++ $CXXFLAGS task1.cpp implementations/lu_mkl.cpp -o task1_mkl \
-I/opt/intel/oneapi/mkl/latest/include \
-L/opt/intel/oneapi/mkl/latest/lib/intel64 \
-lmkl_rt

# Build the Eigen MKL implementation with proper linking
g++ $CXXFLAGS task1.cpp implementations/lu_eigen_mkl.cpp -o task1_eigen_mkl \
-I/opt/intel/oneapi/mkl/latest/include \
-L/opt/intel/oneapi/mkl/latest/lib/intel64 \
-lmkl_rt -lmkl_intel_lp64 -lmkl_sequential -lmkl_core

# Build the cuSolver implementation
g++ $CXXFLAGS task1.cpp implementations/lu_cusolver.cpp -o task1_cusolver


# Make it executable
chmod +x task1_naive
chmod +x task1_mkl
chmod +x task1_eigen_mkl
chmod +x task1_cusolver

echo "You can run task1_naive with: ./task1_naive <matrix_size> <num_threads>"
echo "You can run task1_mkl with: ./task1_mkl <matrix_size> <num_threads>"
echo "You can run task1_eigen_mkl with: ./task1_eigen_mkl <matrix_size> <num_threads>" 
echo "You can run task1_cusolver with: ./task1_cusolver <matrix_size> <num_threads>"
