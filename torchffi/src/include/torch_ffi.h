#ifndef __TORCH_FFI_H__
#define __TORCH_FFI_H__
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#include <torch/torch.h>

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
  int8_t memoryFormat;
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
  void* value;
} Index;

static const uint8_t padModeConstant = 0;
static const uint8_t padModeReflect = 1;
static const uint8_t padModeReplicate = 2;
static const uint8_t padModeCircular = 3;

static const char* padModeNameConstant = "constant";
static const char* padModeNameReflect = "reflect";
static const char* padModeNameReplicate = "replicate";
static const char* padModeNameCircular = "circular";

extern "C" {
typedef torch::Tensor* tensor;
}

#else
typedef void* tensor;
#endif

#ifdef __cplusplus
extern "C" {
#endif
extern tensor torchffi_tensor_new(void);

extern void torchffi_tensor_delete(tensor t);

extern tensor torchffi_tensor_new_zeros(int64_t* sizes, size_t ndims, TensorOptions options);

extern tensor torchffi_tensor_new_ones(int64_t* sizes, size_t ndims, TensorOptions options);

extern tensor torchffi_tensor_new_arange(int64_t end, TensorOptions options);

extern tensor torchffi_tensor_new_rand(int64_t* sizes, size_t ndims, TensorOptions options);

extern size_t torchffi_tensor_dim(tensor t);

extern void torchffi_tensor_sizes(tensor t, size_t dim, int64_t* shape);

extern Device torchffi_tensor_device(tensor t);

extern tensor torchffi_tensor_new_eye(int64_t n, int64_t m, TensorOptions options);

extern tensor torchffi_tensor_new_from_blob(void* data, int64_t* dims, size_t ndims, TensorOptions options);

extern Scalar_t torchffi_tensor_scalar(tensor t);

extern Scalar_t torchffi_tensor_scalar_at(tensor t, int64_t index);

extern tensor torchffi_tensor_get(tensor t, int index);

extern tensor torchffi_tensor_index(tensor t, Index_t* indices, size_t ndims);

extern tensor torchffi_tensor_view(tensor t, int64_t* sizes, size_t ndims);

tensor torchffi_tensor_permute(tensor t, int64_t* dims, size_t ndims);

extern tensor torchffi_tensor_expand(tensor t, int64_t* sizes, size_t ndims, bool implicit);

extern tensor torchffi_tensor_contiguous(tensor t, int8_t memoryFormat);

extern tensor torchffi_tensor_pad(tensor t, int64_t* pad, size_t padArrayLength, uint8_t padMode, double* value);

extern tensor torchffi_tensor_addition(tensor a, tensor b, Scalar alpha);

extern tensor torchffi_tensor_subtraction(tensor a, tensor b, Scalar alpha);

extern tensor torchffi_tensor_multiplication(tensor a, tensor b);

extern tensor torchffi_tensor_division(tensor a, tensor b);

extern tensor torchffi_tensor_bitwise_not(tensor a);

extern tensor torchffi_tensor_bitwise_or(tensor a, tensor b);

extern tensor torchffi_tensor_bitwise_and(tensor a, tensor b);

extern tensor torchffi_tensor_bitwise_xor(tensor a, tensor b);

extern tensor torchffi_tensor_sum(tensor input, int64_t* dim, size_t dimLength, bool keepdim, uint8_t* dtype);

extern tensor torchffi_tensor_mean(tensor input, int64_t* dim, size_t dimLength, bool keepdim, uint8_t* dtype);

extern tensor torchffi_tensor_pow(tensor input, Scalar exponent);

extern tensor torchffi_tensor_rsqrt(tensor input);

extern tensor torchffi_tensor_matmul(tensor a, tensor b);

extern tensor torchffi_tensor_sigmoid(tensor t);

extern tensor torchffi_tensor_gelu(tensor t, char* approximate);

extern tensor torchffi_tensor_silu(tensor t);

extern tensor torchffi_tensor_relu(tensor t);

extern tensor torchffi_linear(tensor input, tensor weight, tensor bias);

extern tensor torchffi_layer_norm(tensor input, int64_t* normalizedShape, size_t normalizedShapeLength, tensor weight, tensor bias, double eps, bool cudnnEnable);

extern tensor torchffi_group_norm(tensor input, int64_t numGroups, tensor weight, tensor bias, float eps);

extern tensor torchffi_tensor_dropout(tensor t, double p, bool train);

extern tensor torchffi_tensor_softmax(tensor t, int64_t dim, uint8_t* dataType);

extern tensor torchffi_embedding(tensor weights, tensor indices, int64_t paddingIdx, uint8_t scaleGradByFreq, uint8_t sparse);

extern tensor torchffi_conv2d(tensor input, tensor weights, tensor bias, int64_t* strides, int64_t* paddings, int64_t* dilations, int64_t groups);

extern tensor torchffi_upsample_nearest(tensor input, int64_t* outputSize, size_t outputSizeLength);

extern tensor torchffi_upsample_nearest_scale(tensor input, double* scales, size_t scalesLength);

extern tensor torchffi_upsample_nearest_exact(tensor input, int64_t* outputSize, size_t outputSizeLength);

extern tensor torchffi_upsample_nearest_exact_scale(tensor input, double* scales, size_t scalesLength);

#ifdef __cplusplus
}
#endif

#endif