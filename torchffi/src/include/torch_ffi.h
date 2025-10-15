#ifndef __TORCH_FFI_H__
#define __TORCH_FFI_H__
#include<stdint.h>
#include <stddef.h>

#ifdef __cplusplus
#include<torch/torch.h>

using namespace std;

typedef enum DataType_t {
    Uint8 = 0,
    Int8 = 1,
    Int16 = 2,
    Int = 3,
    Int64 = 4,
    Half = 5,
    Float = 6,
    Double = 7,
    ComplexHalf = 8,
    ComplexFloat = 9,
    ComplexDouble = 10,
    Bool = 11,
    QInt8 = 12,
    QUInt8 = 13,
    QInt32 = 14,
    BFloat16 = 15,
    Float8e5m2 = 23,
    Float8e4m3fn = 24,
    Float8e5m2fnuz= 25,
    Float8e4m3fnuz = 26,
} DataType;

typedef struct TensorOptions_t {
    // TODO
    // TODO data type DataType 
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

void torchffi_tensor_shape(tensor t, int64_t *dims);

void torchffi_new_tensor_eye(tensor *out__, int64_t n);
#ifdef __cplusplus
}
#endif

#endif