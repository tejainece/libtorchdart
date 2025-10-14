#ifndef __TORCH_FFI_H__
#define __TORCH_FFI_H__
#include<stdint.h>

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
#ifdef __cplusplus
}
#endif

#endif