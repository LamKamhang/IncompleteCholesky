cmake_minimum_required(VERSION 2.6)

CUDA_ADD_EXECUTABLE(target main.cc)
# add_executable(target main.cc)
target_link_libraries(target
  ${CUDA_cusparse_LIBRARY}
)
