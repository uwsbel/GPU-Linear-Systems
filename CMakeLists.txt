cmake_minimum_required(VERSION 3.18)
project(GPU_Linear_Systems LANGUAGES CXX CUDA)

# Set default build type to Release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA architectures for RTX 3070 Ti
set(CMAKE_CUDA_ARCHITECTURES 86)

# Find required packages
find_package(CUDAToolkit REQUIRED)
find_package(Eigen3 REQUIRED)

# Set MKL paths
set(MKL_ROOT "/opt/intel/oneapi/mkl/latest" CACHE PATH "Path to MKL root directory")
set(MKL_INCLUDE_DIR "${MKL_ROOT}/include")
set(MKL_LIB_DIR "${MKL_ROOT}/lib/intel64")

# Define MKL libraries once
set(MKL_LIBRARIES
    ${MKL_LIB_DIR}/libmkl_intel_lp64.so
    ${MKL_LIB_DIR}/libmkl_intel_thread.so
    ${MKL_LIB_DIR}/libmkl_core.so
    -liomp5
)

# Copy data directory to build directory
file(COPY ${CMAKE_SOURCE_DIR}/data DESTINATION ${CMAKE_BINARY_DIR})

# Add executables
add_executable(task_eigen_pardiso task_eigen_pardiso.cpp)
add_executable(task_pardiso task_pardiso.cpp)
add_executable(task_cusolver task_cusolver.cu)
add_executable(task_cudss task_cudss.cu)

# Configure CPU-based targets
foreach(TARGET task_eigen_pardiso task_pardiso)
  target_include_directories(${TARGET} PRIVATE ${MKL_INCLUDE_DIR})
  target_link_libraries(${TARGET} PRIVATE ${MKL_LIBRARIES})
endforeach()

# Add Eigen include directory only to the Eigen target
target_include_directories(task_eigen_pardiso PRIVATE ${EIGEN3_INCLUDE_DIR})

# Configure CUDA target
target_include_directories(task_cusolver PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(task_cusolver PRIVATE
    CUDA::cusolver
    CUDA::cusparse
    CUDA::cublas
    CUDA::cudart
)

# Configure CUDA DSS target
target_include_directories(task_cudss PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(task_cudss PRIVATE
    CUDA::cudss
    CUDA::cusparse
    CUDA::cublas
    CUDA::cudart
)

# Add compile definitions for CUDA version
target_compile_definitions(task_cusolver PRIVATE
    CUDA_VERSION_MAJOR=${CMAKE_CUDA_COMPILER_VERSION_MAJOR}
    CUDA_VERSION_MINOR=${CMAKE_CUDA_COMPILER_VERSION_MINOR}
)

# Add compile definitions for CUDA version to cuDSS target
target_compile_definitions(task_cudss PRIVATE
    CUDA_VERSION_MAJOR=${CMAKE_CUDA_COMPILER_VERSION_MAJOR}
    CUDA_VERSION_MINOR=${CMAKE_CUDA_COMPILER_VERSION_MINOR}
)

# Print configuration summary
message(STATUS "Configuration summary:")
message(STATUS "  CUDA version: ${CMAKE_CUDA_COMPILER_VERSION}")
message(STATUS "  CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")