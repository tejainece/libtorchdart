#include "torch_ffi.h"
#include <torch/torch.h>

extern "C" {

void torchffi_finfo(int8_t dtype, FInfo *info) {
  if (info == nullptr)
    return;

  auto type = static_cast<torch::ScalarType>(dtype);

  try {
    if (type == torch::kFloat32) {
      info->min = std::numeric_limits<float>::lowest();
      info->max = std::numeric_limits<float>::max();
      info->eps = std::numeric_limits<float>::epsilon();
      info->tiny = std::numeric_limits<float>::min();
      info->resolution = 1e-6;
    } else if (type == torch::kFloat64) {
      info->min = std::numeric_limits<double>::lowest();
      info->max = std::numeric_limits<double>::max();
      info->eps = std::numeric_limits<double>::epsilon();
      info->tiny = std::numeric_limits<double>::min();
      info->resolution = 1e-15;
    } else if (type == torch::kHalf) {
      info->min = -65504;
      info->max = 65504;
      info->eps = 0.0009765625;
      info->tiny = 6.103515625e-05;
      info->resolution = 0.001;
    } else if (type == torch::kBFloat16) {
      info->min = -3.38953e38;
      info->max = 3.38953e38;
      info->eps = 0.0078125;
      info->tiny = 1.17549e-38;
      info->resolution = 0.01;
    } else {
      info->min = 0;
      info->max = 0;
      info->eps = 0;
      info->tiny = 0;
      info->resolution = 0;
    }

  } catch (...) {
    info->min = 0;
    info->max = 0;
    info->eps = 0;
    info->tiny = 0;
    info->resolution = 0;
  }
}
}
