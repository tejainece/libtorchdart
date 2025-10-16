#ifndef __TORCH_FFI_H__
#define __TORCH_FFI_H__
#include<stdint.h>
#include <stddef.h>

#ifdef __cplusplus
#include<torch/torch.h>

using namespace std;

typedef struct Device_t {
    int8_t type;
    int8_t index;
} Device;

typedef struct TensorOptions_t {
    int8_t dtype;
    int8_t deviceType;
    int8_t deviceIndex;
    int8_t layout;
    // TODO layout
    // TODO memory format
    // TODO required autograd
    // TODO pinned memory
} TensorOptions;

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

void torchffi_tensor_sizes(tensor t, size_t dim, int64_t *shape);

Device torchffi_tensor_device(tensor t);

tensor torchffi_new_tensor_eye(int64_t n, int64_t m, TensorOptions options);
#ifdef __cplusplus
}
#endif

#endif