# Set the minimum required CMake version and project details
cmake_minimum_required(VERSION 3.18)
project(GPU_Linear_Systems LANGUAGES CXX CUDA)

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# Set CUDA architectures for RTX 3070 Ti
set(CMAKE_CUDA_ARCHITECTURES 86)

# Find required packages
find_package(cudss 0.5.0 REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Eigen3 REQUIRED)

# Set MKL paths
set(MKL_ROOT "/opt/intel/oneapi/mkl/latest" CACHE PATH "Path to MKL root directory")
set(MKL_INCLUDE_DIR "${MKL_ROOT}/include")
set(MKL_LIB_DIR "${MKL_ROOT}/lib/intel64")

# Set IOMP5 library path
set(IOMP5_LIB "/opt/intel/oneapi/compiler/2024.2/lib/libiomp5.so" CACHE FILEPATH "Path to Intel OpenMP Runtime Library (libiomp5.so)")

# Define MKL libraries once
set(MKL_LIBRARIES
    ${MKL_LIB_DIR}/libmkl_intel_lp64.so
    ${MKL_LIB_DIR}/libmkl_intel_thread.so
    ${MKL_LIB_DIR}/libmkl_core.so
    ${IOMP5_LIB}
)

# Include the main project directory to make utils.h available to all targets
include_directories(${CMAKE_SOURCE_DIR})

# Copy data directory to build directory
file(COPY ${CMAKE_SOURCE_DIR}/data DESTINATION ${CMAKE_BINARY_DIR})

# Add executables
# --------------------
# Task Eigen Pardiso
add_executable(task_eigen_pardiso task_eigen_pardiso.cpp)
# Configure CPU-based target for Eigen Pardiso
target_include_directories(task_eigen_pardiso PRIVATE ${MKL_INCLUDE_DIR})
target_link_libraries(task_eigen_pardiso PRIVATE ${MKL_LIBRARIES})
target_include_directories(task_eigen_pardiso PRIVATE ${EIGEN3_INCLUDE_DIR})

# --------------------
# Task Pardiso
add_executable(task_pardiso task_pardiso.cpp)
# Configure CPU-based target for Pardiso
target_include_directories(task_pardiso PRIVATE ${MKL_INCLUDE_DIR})
target_link_libraries(task_pardiso PRIVATE ${MKL_LIBRARIES})

# --------------------
# Task CuSolver
add_executable(task_cusolver task_cusolver.cu)
# Configure CUDA target for CuSolver
target_include_directories(task_cusolver PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(task_cusolver PRIVATE
    CUDA::cusolver
    CUDA::cusparse
)
# Add compile definitions for CUDA version
target_compile_definitions(task_cusolver PRIVATE
    CUDA_VERSION_MAJOR=${CMAKE_CUDA_COMPILER_VERSION_MAJOR}
    CUDA_VERSION_MINOR=${CMAKE_CUDA_COMPILER_VERSION_MINOR}
)

# --------------------
# Task CuDSS
set_source_files_properties(task_simple.cpp PROPERTIES LANGUAGE CUDA)

add_executable(task_simple task_simple.cpp)
# Configure CUDA DSS target
target_include_directories(task_simple PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(task_simple PRIVATE cudss)
# Add compile definitions for CUDA version to cuDSS target
target_compile_definitions(task_simple PRIVATE
    CUDA_VERSION_MAJOR=${CMAKE_CUDA_COMPILER_VERSION_MAJOR}
    CUDA_VERSION_MINOR=${CMAKE_CUDA_COMPILER_VERSION_MINOR}
)


# Print configuration summary
message(STATUS "Configuration summary:")
message(STATUS "  CUDA version: ${CMAKE_CUDA_COMPILER_VERSION}")
message(STATUS "  CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")