#!/bin/bash

# Common compiler flags
CXXFLAGS="-Wall -O3 -std=c++17 -fopenmp"

# Build the naive implementation
g++ $CXXFLAGS task2.cpp implementations/lu_naive.cpp -o task2_naive

# Build the MKL implementation
g++ $CXXFLAGS task2.cpp implementations/lu_mkl.cpp -o task2_mkl \
-I/opt/intel/oneapi/mkl/latest/include \
-L/opt/intel/oneapi/mkl/latest/lib/intel64 \
-lmkl_rt

# Build the Eigen with MKL implementation
g++ $CXXFLAGS task2.cpp implementations/lu_eigen_mkl.cpp -o task2_eigen_mkl \
-I/usr/include/eigen3 \
-I/opt/intel/oneapi/mkl/latest/include \
-L/opt/intel/oneapi/mkl/latest/lib/intel64 \
-lmkl_rt \
-DEIGEN_USE_MKL_ALL

# Build the CUDA cuSolver implementation
nvcc -O3 -std=c++17 implementations/lu_cusolver.cpp task2.cpp -o task2_cusolver \
-Xcompiler=-fopenmp \
-I/usr/local/cuda/include \
-lcudart -lcusolver -lcublas

if [ $? -eq 0 ]; then
    echo "Build successful!"
else
    echo "Build failed!"
    exit 1
fi

# Make it executable
chmod +x task2_naive
chmod +x task2_mkl
chmod +x task2_eigen_mkl
chmod +x task2_cusolver

echo "You can run task2_naive with: ./task2_naive <num_threads>"
echo "You can run task2_mkl with: ./task2_mkl <num_threads>"
echo "You can run task2_eigen_mkl with: ./task2_eigen_mkl <num_threads>"
echo "You can run task2_cusolver with: ./task2_cusolver <num_threads>" 