cmake_minimum_required(VERSION 2.6)

project(IncompleteCholesky)

FIND_PACKAGE(CUDA ${CUDA_VERSION} REQUIRED)
if (CUDA_FOUND)
  SET(CUDA_NVCC_FLAGS --compiler-options "-std=c++11")
  SET(CUDA_HELPER_PATH ${PROJECT_SOURCE_DIR}/external/cuda-samples/Common)
  message(STATUS "cuda_helper_path: ${CUDA_HELPER_PATH}")
  message(STATUS "cuda_include_dirs: ${CUDA_INCLUDE_DIRS}")
  include_directories(${CUDA_HELPER_PATH})
  include_directories(${CUDA_INCLUDE_DIRS})
endif(CUDA_FOUND)

find_package(Eigen3 REQUIRED)
if(EIGEN3_FOUND)
  message(STATUS "Eigen_include_dirs: ${EIGEN3_INCLUDE_DIRS}")
  include_directories(${EIGEN3_INCLUDE_DIRS})
endif(EIGEN3_FOUND)

add_subdirectory(src)
add_subdirectory(tests)
