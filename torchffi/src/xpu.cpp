#include <torch/all.h>
#include <torch_ffi.h>

#if defined(WITH_XPU)
#include <c10/xpu/XPUCachingAllocator.h>
#endif

bool torchffi_is_xpu_available() { return torch::xpu::is_available(); }

int64_t torchffi_xpu_memory_total(int deviceIndex) {
#if defined(WITH_XPU)
  if (deviceIndex >= torch::xpu::device_count()) {
    return 0;
  }
  torch::xpu::DeviceProp props;
  torch::xpu::get_device_properties(&props, deviceIndex);
  return props.global_mem_size;
#else
  return 0;
#endif
}

int64_t torchffi_xpu_memory_allocated(int deviceIndex, char **error) {
#if defined(WITH_XPU)
  if (!torch::xpu::is_available()) {
    return 0;
  }
  try {
    at::globalContext().lazyInitDevice(at::kXPU);
    if (deviceIndex == -1) {
      deviceIndex = at::xpu::current_device();
    }
    // c10::xpu::XPUGuard guard(deviceIndex);
    at::xpu::set_device(deviceIndex);
    auto allocations =
        c10::xpu::XPUCachingAllocator::getDeviceStats(deviceIndex)
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

int64_t torchffi_xpu_memory_reserved(int deviceIndex, char **error) {
#if defined(WITH_XPU)
  if (!torch::xpu::is_available()) {
    return 0;
  }
  try {
    if (deviceIndex == -1) {
      deviceIndex = at::xpu::current_device();
    }
    auto reservations =
        c10::xpu::XPUCachingAllocator::getDeviceStats(deviceIndex)
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

int64_t torchffi_xpu_device_count() {
#if defined(WITH_XPU)
  return torch::xpu::device_count();
#else
  return 0;
#endif
}
