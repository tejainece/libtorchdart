#include <torch/all.h>
#include <torch_ffi.h>

#include <ATen/mps/MPSAllocatorInterface.h>

#ifdef __cplusplus
extern "C" {
#endif

bool torchffi_is_mps_available() { return torch::mps::is_available(); }

int64_t torchffi_mps_current_allocated_memory() {
  if (!torch::mps::is_available())
    return 0;
  return at::mps::getIMPSAllocator()->getCurrentAllocatedMemory();
}

int64_t torchffi_mps_driver_allocated_memory() {
  if (!torch::mps::is_available())
    return 0;
  return at::mps::getIMPSAllocator()->getDriverAllocatedMemory();
}

int64_t torchffi_mps_recommended_max_memory() {
  if (!torch::mps::is_available())
    return 0;
  return at::mps::getIMPSAllocator()->getRecommendedMaxMemory();
}

#ifdef __cplusplus
}
#endif