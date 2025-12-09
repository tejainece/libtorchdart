#include <torch/all.h>
#include <torch_ffi.h>

#if defined(WITH_CUDA)
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAContextLight.h>
#include <c10/core/DeviceType.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

bool torchffi_is_cuda_available() {
#if defined(WITH_CUDA)
  return torch::cuda::is_available();
#else
  return false;
#endif
}

CDeviceProperties *torchffi_cuda_get_device_properties(int deviceIndex) {
#if defined(WITH_CUDA)
  cudaDeviceProp *prop = at::cuda::getDeviceProperties(deviceIndex);
  CDeviceProperties *out =
      (CDeviceProperties *)malloc(sizeof(CDeviceProperties));
  out->name = strdup(prop->name);
  out->totalMemory = prop->totalGlobalMem;
  out->multiProcessorCount = prop->multiProcessorCount;
  out->major = prop->major;
  out->minor = prop->minor;
  return out;
#else
  return nullptr;
#endif
}

int64_t torchffi_cuda_memory_total(int deviceIndex) {
#if defined(WITH_CUDA)
  if (deviceIndex == -1) {
    deviceIndex = at::cuda::current_device();
  }
  return at::cuda::getDeviceProperties(deviceIndex)->totalGlobalMem;
#else
  return 0;
#endif
}

int64_t torchffi_cuda_memory_allocated(int8_t deviceIndex, char **error) {
#if defined(WITH_CUDA)
  if (!torch::cuda::is_available()) {
    return 0;
  }
  try {
    at::globalContext().lazyInitDevice(at::kCUDA);
    if (deviceIndex == -1) {
      deviceIndex = at::cuda::current_device();
    }
    // TODO c10::cuda::CUDAGuard guard(deviceIndex);
    // TODO at::cuda::set_device(deviceIndex);
    auto allocations =
        c10::cuda::CUDACachingAllocator::getDeviceStats(deviceIndex)
            .allocated_bytes;
    int64_t allocated = 0;
    for (auto allocation : allocations) {
      allocated += allocation.current;
    }
    return allocated;
  } catch (const c10::Error &e) {
    *error = strdup(e.what());
    return 0;
  }
#else
  return 0;
#endif
}

int64_t torchffi_cuda_memory_reserved(int8_t deviceIndex, char **error) {
#if defined(WITH_CUDA)
  if (!torch::cuda::is_available()) {
    return 0;
  }
  try {
    if (deviceIndex == -1) {
      deviceIndex = at::cuda::current_device();
    }
    auto reservations =
        c10::cuda::CUDACachingAllocator::getDeviceStats(deviceIndex)
            .reserved_bytes;
    int64_t reserved = 0;
    for (auto reservation : reservations) {
      reserved += reservation.current;
    }
    return reserved;
  } catch (const c10::Error &e) {
    *error = strdup(e.what());
    return 0;
  }
#else
  return 0;
#endif
}

int64_t torchffi_cuda_device_count() {
#if defined(WITH_CUDA)
  if (!torch::cuda::is_available()) {
    return 0;
  }
  return torch::cuda::device_count();
#else
  return 0;
#endif
}

#ifdef __cplusplus
}
#endif