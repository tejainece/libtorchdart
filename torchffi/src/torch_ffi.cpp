#include <torch/all.h>
#include <torch_ffi.h>

tensor torchffi_new_tensor() {
    return new torch::Tensor();
}

size_t torchffi_tensor_dim(tensor t) {
    return t->dim();
}

void torchffi_tensor_shape(tensor t, int64_t *dims) {
    int i = 0;
    for (int64_t dim : t->sizes()) dims[i++] = dim;
}

at::Device device_of_int(int d) {
    if (d == -3) return at::Device(at::kVulkan);
    if (d == -2) return at::Device(at::kMPS);
    if (d < 0) return at::Device(at::kCPU);
    return at::Device(at::kCUDA, /*index=*/d);
}

void torchffi_new_tensor_eye(tensor *out__, int64_t n, int options_kind, int options_device) {
    at::Tensor outputs__ = torch::eye(n, at::device(device_of_int(options_device)).dtype(at::ScalarType(options_kind)));
    out__[0] = new torch::Tensor(outputs__);
}