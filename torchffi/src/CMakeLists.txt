cmake_minimum_required(VERSION 3.31.6)
project(torchffi_library)
#set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.9/bin/nvcc)
#set(CMAKE_CUDA_ARCHITECTURES native)

set(libtorch_VERSION 2.8.0)
find_package(Torch ${libtorch_VERSION} PATHS ../libtorch)
if(NOT Torch_FOUND)
  # TODO
else()
  message(STATUS "libtorch ${libtorch_VERSION} - found")
endif()

add_library(torchffi SHARED torch_ffi.cpp)
target_include_directories(torchffi PRIVATE ./include)
target_include_directories(torchffi PRIVATE ../libtorch/include)
target_include_directories(torchffi PRIVATE ../libtorch/include/torch/csrc/api/include)
target_link_libraries(torchffi ${TORCH_LIBRARIES})
