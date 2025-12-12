cmake_minimum_required(VERSION 3.31.2)
project(torchffi_library)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-13/bin/nvcc)
set(CMAKE_CUDA_ARCHITECTURES all)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_C_COMPILER "clang")
set(CMAKE_CXX_COMPILER "clang++")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(libtorch_VERSION 2.8.0)
find_package(Torch ${libtorch_VERSION} PATHS ./libtorch)
if(NOT Torch_FOUND)
  # TODO
else()
  message(STATUS "libtorch ${libtorch_VERSION} - found")
endif()
# Define WITH_CUDA for non-mac platforms
if(NOT APPLE)
  add_definitions(-DWITH_CUDA)
endif()

add_library(torchffi SHARED src/tensor.cpp src/generator.cpp src/cuda.cpp src/mps.cpp src/xpu.cpp src/finfo.cpp)
target_include_directories(torchffi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(torchffi PRIVATE ./libtorch/include)
target_include_directories(torchffi PRIVATE ./libtorch/include/torch/csrc/api/include)
target_compile_features(torchffi PRIVATE cxx_std_17)
target_link_libraries(torchffi ${TORCH_LIBRARIES})
