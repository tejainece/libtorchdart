cmake_minimum_required(VERSION 3.31.2)
project(torchffi_library)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-13/bin/nvcc)
set(CMAKE_CUDA_ARCHITECTURES all)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_C_COMPILER "clang")
set(CMAKE_CXX_COMPILER "clang++")

set(libtorch_VERSION 2.8.0)
find_package(Torch ${libtorch_VERSION} PATHS ../libtorch)
if(NOT Torch_FOUND)
  # TODO
else()
  message(STATUS "libtorch ${libtorch_VERSION} - found")
endif()

add_library(torchffi SHARED torch_ffi.cpp generator.cpp)
target_include_directories(torchffi PRIVATE ./include)
target_include_directories(torchffi PRIVATE ../libtorch/include)
target_include_directories(torchffi PRIVATE ../libtorch/include/torch/csrc/api/include)
target_compile_features(torchffi PRIVATE cxx_std_17)
target_link_libraries(torchffi ${TORCH_LIBRARIES})
