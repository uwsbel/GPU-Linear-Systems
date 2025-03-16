cmake_minimum_required(VERSION 3.10)
project(GPU_Linear_Systems)

# Set default build type to Release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MKL
set(MKL_ROOT "/opt/intel/oneapi/mkl/latest" CACHE PATH "Path to MKL root directory")
set(MKL_INCLUDE_DIR "${MKL_ROOT}/include")
set(MKL_LIB_DIR "${MKL_ROOT}/lib/intel64")

# Copy data directory to build directory
file(COPY ${CMAKE_SOURCE_DIR}/data DESTINATION ${CMAKE_BINARY_DIR})

# Find Eigen
find_package(Eigen3 REQUIRED)

# Add executables
add_executable(task_eigen_pardiso task_eigen_pardiso.cpp)
add_executable(task_pardiso task_pardiso.cpp)

# Include directories
target_include_directories(task_eigen_pardiso PRIVATE 
    ${MKL_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIR}
)

target_include_directories(task_pardiso PRIVATE 
    ${MKL_INCLUDE_DIR}
)

# Link libraries
target_link_libraries(task_eigen_pardiso PRIVATE
    ${MKL_LIB_DIR}/libmkl_intel_lp64.so
    ${MKL_LIB_DIR}/libmkl_intel_thread.so
    ${MKL_LIB_DIR}/libmkl_core.so
    -liomp5
) 

target_link_libraries(task_pardiso PRIVATE
    ${MKL_LIB_DIR}/libmkl_intel_lp64.so
    ${MKL_LIB_DIR}/libmkl_intel_thread.so
    ${MKL_LIB_DIR}/libmkl_core.so
    -liomp5
) 