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
find_package(cudss 0.7.1 REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Eigen3 REQUIRED)

# Set MKL paths
set(MKL_ROOT "/opt/intel/oneapi/mkl/latest" CACHE PATH "Path to MKL root directory")
set(MKL_INCLUDE_DIR "${MKL_ROOT}/include")
set(MKL_LIB_DIR "${MKL_ROOT}/lib/intel64")

# Set IOMP5 library path
set(IOMP5_LIB "/opt/intel/oneapi/compiler/2025.0/lib/libiomp5.so" CACHE FILEPATH "Path to Intel OpenMP Runtime Library (libiomp5.so)")

# Define MKL libraries once
set(MKL_LIBRARIES
    ${MKL_LIB_DIR}/libmkl_intel_lp64.so
    ${MKL_LIB_DIR}/libmkl_intel_thread.so
    ${MKL_LIB_DIR}/libmkl_core.so
    ${IOMP5_LIB}
)

set(PROJECT_INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/common
    ${CMAKE_SOURCE_DIR}/solvers
)

# Copy data directory to build directory
file(COPY ${CMAKE_SOURCE_DIR}/data DESTINATION ${CMAKE_BINARY_DIR})

function(add_mkl_driver target_name source_file)
  add_executable(${target_name} ${source_file})
  target_include_directories(${target_name} PRIVATE ${PROJECT_INCLUDE_DIRS} ${MKL_INCLUDE_DIR})
  target_link_libraries(${target_name} PRIVATE ${MKL_LIBRARIES})
endfunction()

function(add_cudss_driver target_name source_file)
  set_source_files_properties(${source_file} PROPERTIES LANGUAGE CUDA)
  add_executable(${target_name} ${source_file})
  target_include_directories(${target_name} PRIVATE ${PROJECT_INCLUDE_DIRS} ${CUDAToolkit_INCLUDE_DIRS})
  target_link_libraries(${target_name} PRIVATE cudss)
  target_compile_definitions(${target_name} PRIVATE
      CUDA_VERSION_MAJOR=${CMAKE_CUDA_COMPILER_VERSION_MAJOR}
      CUDA_VERSION_MINOR=${CMAKE_CUDA_COMPILER_VERSION_MINOR}
  )
endfunction()

# Add executables
add_mkl_driver(run_eigen_pardiso drivers/run_eigen_pardiso.cpp)
target_include_directories(run_eigen_pardiso PRIVATE ${EIGEN3_INCLUDE_DIR})

add_mkl_driver(run_pardiso drivers/run_pardiso.cpp)
add_cudss_driver(run_cudss drivers/run_cudss.cpp)


# Print configuration summary
message(STATUS "Configuration summary:")
message(STATUS "  CUDA version: ${CMAKE_CUDA_COMPILER_VERSION}")
message(STATUS "  CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
