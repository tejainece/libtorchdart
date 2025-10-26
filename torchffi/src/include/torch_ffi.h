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

typedef struct Slice_t {
    int64_t* start;
    int64_t* stop;
    int64_t step;
} Slice;

typedef struct Index_t {
    uint8_t type;
    void *value;
} Index;

extern "C" {
typedef torch::Tensor *tensor;
}

#else
typedef void *tensor;
#endif

#ifdef __cplusplus
extern "C" {
#endif
extern tensor torchffi_new_tensor(void);

extern tensor torchffi_tensor_new_zeros(int64_t *sizes, size_t ndims, TensorOptions options);

extern tensor torchffi_tensor_new_ones(int64_t *sizes, size_t ndims, TensorOptions options);

extern tensor torchffi_tensor_new_arange(int64_t end, TensorOptions options);

extern tensor torchffi_tensor_new_rand(int64_t *sizes, size_t ndims, TensorOptions options);

extern size_t torchffi_tensor_dim(tensor t);

extern void torchffi_tensor_sizes(tensor t, size_t dim, int64_t *shape);

extern Device torchffi_tensor_device(tensor t);

extern tensor torchffi_tensor_new_eye(int64_t n, int64_t m, TensorOptions options);

extern tensor torchffi_tensor_new_from_blob(void *data, int64_t *dims, size_t ndims, TensorOptions options);

extern Scalar_t torchffi_tensor_item(tensor t);

extern tensor torchffi_tensor_get(tensor t, int index);

extern tensor torchffi_tensor_index(tensor t, Index_t* indices, size_t ndims);

extern tensor torchffi_tensor_view(tensor t, int64_t *sizes, size_t ndims);

extern tensor torchffi_tensor_expand(tensor t, int64_t *sizes, size_t ndims, bool implicit);

extern tensor torchffi_tensor_addition(tensor a, tensor b, Scalar alpha);

extern tensor torchffi_tensor_subtraction(tensor a, tensor b, Scalar alpha);

extern tensor torchffi_tensor_multiplication(tensor a, tensor b);

extern tensor torchffi_tensor_division(tensor a, tensor b);

extern tensor torchffi_tensor_sigmoid(tensor t);

extern tensor torchffi_tensor_gelu(tensor t, char* approximate);

extern tensor torchffi_linear(tensor input, tensor weight, tensor bias);

extern tensor torchffi_layer_norm(tensor input, int64_t* normalizedShape, size_t normalizedShapeLength, tensor weight, tensor bias, double eps, bool cudnnEnable);

extern tensor torchffi_embedding(tensor weights, tensor indices, int64_t paddingIdx, uint8_t scaleGradByFreq, uint8_t sparse);

#ifdef __cplusplus
}
#endif

#endif