#ifndef __TORCH_FFI_H__
#define __TORCH_FFI_H__
#include<stdint.h>
#include <stddef.h>

#ifdef __cplusplus
#include<torch/torch.h>

using namespace std;

extern "C" {
typedef torch::Tensor *tensor;
typedef torch::Scalar *scalar;
}

#else
typedef void *tensor;
typedef void *scalar;
#endif

#ifdef __cplusplus
extern "C" {
#endif
tensor torchffi_new_tensor(void);

size_t torchffi_tensor_dim(tensor t);

void torchffi_tensor_shape(tensor t, int64_t *dims);

void torchffi_new_tensor_eye(tensor *out__, int64_t n, int options_kind, int options_device);
#ifdef __cplusplus
}
#endif

#endif