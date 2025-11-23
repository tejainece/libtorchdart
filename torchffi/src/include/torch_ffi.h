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
  int8_t *dtype;
  Device *device;
  int8_t *layout;
  int8_t *memoryFormat;
  bool *requiresGrad;
  bool *pinnedMemory;
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
  int64_t *start;
  int64_t *stop;
  int64_t step;
} Slice;

typedef struct Index_t {
  uint8_t type;
  void *value;
} Index;

static const uint8_t padModeConstant = 0;
static const uint8_t padModeReflect = 1;
static const uint8_t padModeReplicate = 2;
static const uint8_t padModeCircular = 3;

static const char *padModeNameConstant = "constant";
static const char *padModeNameReflect = "reflect";
static const char *padModeNameReplicate = "replicate";
static const char *padModeNameCircular = "circular";

extern "C" {
typedef torch::Tensor *tensor;
typedef torch::Generator *Generator;
}

#else
typedef void *tensor;
typedef void *Generator;
#endif

#ifdef __cplusplus
extern "C" {
#endif
extern tensor torchffi_tensor_new(void);

extern void torchffi_tensor_delete(tensor t);

extern tensor torchffi_tensor_clone(tensor t, int8_t *memoryFormat);

extern tensor torchffi_tensor_new_empty(int64_t *sizes, size_t ndims,
                                        TensorOptions options);

extern tensor torchffi_tensor_new_zeros(int64_t *sizes, size_t ndims,
                                        TensorOptions options);

extern tensor torchffi_tensor_new_ones(int64_t *sizes, size_t ndims,
                                       TensorOptions options);

extern tensor torchffi_tensor_new_arange(int64_t end, TensorOptions options);

extern tensor torchffi_tensor_new_rand(int64_t *sizes, size_t ndims,
                                       Generator generator,
                                       TensorOptions options);

extern tensor torchffi_tensor_new_randn(int64_t *sizes, size_t ndims,
                                        Generator generator,
                                        TensorOptions options);

extern void *torchffi_tensor_data_pointer(tensor t);

extern size_t torchffi_tensor_dim(tensor t);

extern void torchffi_tensor_sizes(tensor t, size_t dim, int64_t *shape);

extern Device torchffi_tensor_device(tensor t);

extern tensor torchffi_tensor_new_eye(int64_t n, int64_t m,
                                      TensorOptions options);

extern tensor torchffi_tensor_new_from_blob(void *data, int64_t *dims,
                                            size_t ndims,
                                            TensorOptions options);

extern Scalar_t torchffi_tensor_scalar(tensor t);

extern Scalar_t torchffi_tensor_scalar_at(tensor t, int64_t index);

extern tensor torchffi_tensor_get(tensor t, int index);

extern int8_t torchffi_tensor_get_datatype(tensor t);

extern tensor torchffi_tensor_to(tensor t, TensorOptions options,
                                 bool nonBlocking, bool copy);

extern tensor torchffi_tensor_index(tensor t, Index_t *indices, size_t ndims);

extern tensor torchffi_tensor_view(tensor t, int64_t *sizes, size_t ndims);

extern tensor torchffi_tensor_reshape(tensor t, int64_t *sizes, size_t ndims);

extern tensor torchffi_tensor_flatten(tensor t, int64_t startDim,
                                      int64_t endDim);

extern tensor *torchffi_tensor_split_equally(tensor t, int64_t splits,
                                             int64_t dim);

extern tensor *torchffi_tensor_split(tensor t, int64_t *splits,
                                     size_t splitsSize, int64_t dim);

extern tensor *torchffi_tensor_chunk(tensor t, int64_t chunks, int64_t dim);

extern tensor torchffi_tensor_permute(tensor t, int64_t *dims, size_t ndims);

extern tensor torchffi_tensor_expand(tensor t, int64_t *sizes, size_t ndims,
                                     bool implicit);

extern tensor torchffi_tensor_contiguous(tensor t, int8_t memoryFormat);

extern tensor torchffi_tensor_squeeze(tensor t, int64_t *dim);

extern tensor torchffi_tensor_unsqueeze(tensor t, int64_t dim);

extern tensor torchffi_tensor_pad(tensor t, int64_t *pad, size_t padArrayLength,
                                  uint8_t padMode, double *value);

extern void torchffi_tensor_ones_(tensor t);

extern void torchffi_tensor_zeros_(tensor t);

extern void torchffi_tensor_eye_(tensor t);

extern void torchffi_tensor_fill_(tensor t, Scalar value);

extern void torchffi_tensor_rand_(tensor t, Generator generator);

extern void torchffi_tensor_normal_(tensor t, Generator generator, double mean,
                                    double std);

extern void torchffi_tensor_uniform_(tensor t, Generator generator, double from,
                                     double to);

extern bool torchffi_tensor_allclose(tensor a, tensor b, double rtol,
                                     double atol, bool equalNan);

extern tensor torchffi_tensor_addition(tensor a, tensor b, Scalar alpha);

extern tensor torchffi_tensor_subtraction(tensor a, tensor b, Scalar alpha);

extern tensor torchffi_tensor_multiplication(tensor a, tensor b);

extern tensor torchffi_tensor_division(tensor a, tensor b);

extern tensor torchffi_tensor_division_scalar(tensor a, Scalar b);

extern tensor torchffi_tensor_bitwise_not(tensor a);

extern tensor torchffi_tensor_bitwise_or(tensor a, tensor b);

extern tensor torchffi_tensor_bitwise_and(tensor a, tensor b);

extern tensor torchffi_tensor_bitwise_xor(tensor a, tensor b);

extern tensor torchffi_tensor_sum(tensor input, int64_t *dim, size_t dimLength,
                                  bool keepdim, uint8_t *dtype);

extern tensor torchffi_tensor_mean(tensor input, int64_t *dim, size_t dimLength,
                                   bool keepdim, uint8_t *dtype);

extern tensor torchffi_tensor_pow(tensor input, Scalar exponent);

extern tensor torchffi_tensor_rsqrt(tensor input);

extern tensor torchffi_tensor_sin(tensor input);

extern tensor torchffi_tensor_cos(tensor input);

extern tensor torchffi_tensor_exp(tensor input);

extern tensor torchffi_tensor_matmul(tensor a, tensor b);

extern tensor torchffi_tensor_sigmoid(tensor t);

extern tensor torchffi_tensor_gelu(tensor t, char *approximate);

extern tensor torchffi_tensor_silu(tensor t);

extern tensor torchffi_tensor_relu(tensor t);

extern tensor torchffi_linear(tensor input, tensor weight, tensor bias);

extern tensor torchffi_layer_norm(tensor input, int64_t *normalizedShape,
                                  size_t normalizedShapeLength, tensor weight,
                                  tensor bias, double eps, bool cudnnEnable);

extern tensor torchffi_group_norm(tensor input, int64_t numGroups,
                                  tensor weight, tensor bias, float eps);

extern tensor torchffi_rms_norm(tensor input, int64_t *normalizedShape,
                                size_t normalizedShapeLength, tensor weight,
                                double *eps);

extern tensor torchffi_dropout(tensor t, double p, bool train);

extern void torchffi_dropout_(tensor t, double p, bool train);

extern tensor torchffi_tensor_softmax(tensor t, int64_t dim, uint8_t *dataType);

extern tensor torchffi_embedding_renorm_(tensor weights, tensor indices,
                                         double maxNorm, double normType);

extern tensor torchffi_embedding(tensor weights, tensor indices,
                                 int64_t paddingIdx, uint8_t scaleGradByFreq,
                                 uint8_t sparse);

extern tensor torchffi_conv2d(tensor input, tensor weights, tensor bias,
                              int64_t *strides, int64_t *paddings,
                              int64_t *dilations, int64_t groups);

extern tensor torchffi_upsample_nearest(tensor input, int64_t *outputSize,
                                        size_t outputSizeLength);

extern tensor torchffi_upsample_nearest_scale(tensor input, double *scales,
                                              size_t scalesLength);

extern tensor torchffi_upsample_nearest_exact(tensor input, int64_t *outputSize,
                                              size_t outputSizeLength);

extern tensor torchffi_upsample_nearest_exact_scale(tensor input,
                                                    double *scales,
                                                    size_t scalesLength);

extern tensor torchffi_avg_pool2d(tensor input, int64_t kernelSizeH,
                                  int64_t kernelSizeW, int64_t strideH,
                                  int64_t strideW, int64_t paddingH,
                                  int64_t paddingW, bool ceilMode,
                                  bool countIncludePad,
                                  int64_t *divisorOverride);

// Generators

extern Generator torchffi_get_default_generator(Device *device);

extern Generator torchffi_generator_new();

extern Generator torchffi_generator_clone(Generator generator);

extern void torchffi_generator_delete(Generator generator);

extern void torchffi_generator_set_current_seed(Generator generator,
                                                uint64_t seed);

extern uint64_t torchffi_generator_get_current_seed(Generator generator);

extern void torchffi_generator_set_offset(Generator generator, uint64_t offset);

extern uint64_t torchffi_generator_get_offset(Generator generator);

extern void torchffi_generator_set_state(Generator generator, tensor newState);

extern tensor torchffi_generator_get_state(Generator generator);

extern Device torchffi_generator_get_device(Generator generator);

#ifdef __cplusplus
}
#endif

#endif