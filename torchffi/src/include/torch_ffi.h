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
    // TODO memory format
    // TODO required autograd
    // TODO pinned memory
} TensorOptions;

typedef struct Scalar_t {
    int8_t dtype;
    union {
        bool b;
        int64_t i;
        double d;
    } value;
} Scalar;

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
extern tensor torchffi_new_tensor(void);

extern size_t torchffi_tensor_dim(tensor t);

extern void torchffi_tensor_sizes(tensor t, size_t dim, int64_t *shape);

extern Device torchffi_tensor_device(tensor t);

extern tensor torchffi_new_tensor_eye(int64_t n, int64_t m, TensorOptions options);

extern tensor torchffi_tensor_new_from_blob(void *data, int64_t *dims, size_t ndims, TensorOptions options);

extern Scalar_t torchffi_tensor_item(tensor t);

extern tensor torchffi_tensor_get(tensor t, int index);
#ifdef __cplusplus
}
#endif

#endif