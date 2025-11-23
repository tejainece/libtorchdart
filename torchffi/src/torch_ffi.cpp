#include <stdlib.h>
#include <torch/all.h>
#include <torch_ffi.h>

#include <optional>

at::TensorOptions torchffi_make_tensor_options(TensorOptions options) {
  at::TensorOptions tensorOptions;
  if (options.dtype != nullptr) {
    tensorOptions = tensorOptions.dtype(at::ScalarType(*options.dtype));
  }
  if (options.device != nullptr) {
    tensorOptions = tensorOptions.device(at::Device(
        at::DeviceType(options.device->type), options.device->index));
  }
  if (options.layout != nullptr) {
    tensorOptions = tensorOptions.layout(at::Layout(*options.layout));
  }
  if (options.memoryFormat != nullptr) {
    tensorOptions =
        tensorOptions.memory_format(at::MemoryFormat(*options.memoryFormat));
  }
  if (options.requiresGrad != nullptr) {
    tensorOptions = tensorOptions.requires_grad(*options.requiresGrad);
  }
  if (options.pinnedMemory != nullptr) {
    tensorOptions = tensorOptions.pinned_memory(*options.pinnedMemory);
  }
  return tensorOptions;
}

at::indexing::TensorIndex torchffi_make_tensor_index(Index_t &index) {
  at::indexing::TensorIndexType type =
      (at::indexing::TensorIndexType)index.type;
  if (type == at::indexing::TensorIndexType::None) {
    return at::indexing::None;
  } else if (type == at::indexing::TensorIndexType::Ellipsis) {
    return at::indexing::Ellipsis;
  } else if (type == at::indexing::TensorIndexType::SymInt) {
    return at::indexing::TensorIndex(*(int64_t *)index.value);
  } else if (type == at::indexing::TensorIndexType::Boolean) {
    return at::indexing::TensorIndex(*(bool *)index.value);
  } else if (type == at::indexing::TensorIndexType::Slice) {
    Slice slice = *(Slice *)index.value;
    return at::indexing::TensorIndex(at::indexing::Slice(
        slice.start ? std::make_optional(c10::SymInt(*slice.start))
                    : std::nullopt,
        slice.stop ? std::make_optional(c10::SymInt(*slice.stop))
                   : std::nullopt,
        c10::SymInt(slice.step)));
  } else if (type == at::indexing::TensorIndexType::Tensor) {
    return at::indexing::TensorIndex(*(torch::Tensor *)index.value);
  } else {
    return at::indexing::Ellipsis;
  }
}

at::Scalar torchffi_to_scalar(Scalar alpha) {
  at::Scalar opAlpha = at::Scalar(0);
  if (alpha.dtype == 0) {
    opAlpha = at::Scalar(alpha.value.b);
  } else if (alpha.dtype == 1) {
    opAlpha = at::Scalar(alpha.value.i);
  } else if (alpha.dtype == 2) {
    opAlpha = at::Scalar(alpha.value.d);
  }
  return opAlpha;
}

void torchffi_tensor_delete(tensor t) { delete t; }

tensor torchffi_tensor_new() { return new torch::Tensor(); }

tensor torchffi_tensor_clone(tensor t, int8_t *memoryFormat) {
  std::optional<at::MemoryFormat> format = std::nullopt;
  if (memoryFormat != nullptr) {
    format = at::MemoryFormat(*memoryFormat);
  }
  at::Tensor tensor = t->clone(format);
  return new torch::Tensor(tensor);
}

tensor torchffi_tensor_new_empty(int64_t *sizes, size_t ndims,
                                 TensorOptions options) {
  at::Tensor tensor = at::empty(at::IntArrayRef(sizes, ndims),
                                torchffi_make_tensor_options(options));
  return new torch::Tensor(tensor);
}

tensor torchffi_tensor_new_zeros(int64_t *sizes, size_t ndims,
                                 TensorOptions options) {
  at::Tensor tensor = at::zeros(at::IntArrayRef(sizes, ndims),
                                torchffi_make_tensor_options(options));
  return new torch::Tensor(tensor);
}

tensor torchffi_tensor_new_ones(int64_t *sizes, size_t ndims,
                                TensorOptions options) {
  at::Tensor tensor = at::ones(at::IntArrayRef(sizes, ndims),
                               torchffi_make_tensor_options(options));
  return new torch::Tensor(tensor);
}

tensor torchffi_tensor_new_arange(int64_t end, TensorOptions options) {
  at::Tensor tensor = at::arange(end, torchffi_make_tensor_options(options));
  return new torch::Tensor(tensor);
}

tensor torchffi_tensor_new_rand(int64_t *sizes, size_t ndims,
                                Generator generator, TensorOptions options) {
  at::Tensor tensor = at::rand(
      at::IntArrayRef(sizes, ndims),
      generator ? std::optional<at::Generator>(*generator) : std::nullopt,
      torchffi_make_tensor_options(options));
  return new torch::Tensor(tensor);
}

tensor torchffi_tensor_new_randn(int64_t *sizes, size_t ndims,
                                 Generator generator, TensorOptions options) {
  at::Tensor tensor = at::randn(
      at::IntArrayRef(sizes, ndims),
      generator ? std::optional<at::Generator>(*generator) : std::nullopt,
      torchffi_make_tensor_options(options));
  return new torch::Tensor(tensor);
}

tensor torchffi_tensor_new_eye(int64_t n, int64_t m, TensorOptions options) {
  at::Tensor tensor = at::eye(n, m, torchffi_make_tensor_options(options));
  return new torch::Tensor(tensor);
}

tensor torchffi_tensor_new_from_blob(void *data, int64_t *dims, size_t ndims,
                                     TensorOptions options) {
  at::Tensor tensor = at::from_blob(data, torch::IntArrayRef(dims, ndims),
                                    torchffi_make_tensor_options(options));
  return new torch::Tensor(tensor);
}

void *torchffi_tensor_data_pointer(tensor t) { return t->data_ptr(); }

size_t torchffi_tensor_dim(tensor t) { return t->dim(); }

void torchffi_tensor_sizes(tensor t, size_t dim, int64_t *shape) {
  int i = 0;
  for (int64_t dimShape : t->sizes()) {
    if (i == dim) {
      break;
    }
    shape[i++] = dimShape;
  }
  for (; i < dim; i++) {
    shape[i] = 0;
  }
}

Device torchffi_tensor_device(tensor t) {
  auto device = t->device();
  return Device{int8_t(device.type()), device.index()};
}

Scalar torchffi_tensor_scalar(tensor t) {
  at::Scalar scalar = t->item();
  at::ScalarType type = scalar.type();
  if (scalar.isBoolean()) {
    return {
        .dtype = 0,
        .value = {.b = scalar.toBool()},
    };
  } else if (scalar.isIntegral(false)) {
    return {
        .dtype = 1,
        .value = {.i = scalar.toLong()},
    };
  } else if (scalar.isFloatingPoint()) {
    return {
        .dtype = 2,
        .value = {.d = scalar.toDouble()},
    };
  } else {
    return {
        .dtype = -1,
    };
  }
}

Scalar_t torchffi_tensor_scalar_at(tensor t, int64_t index) {
  auto tensor = torch::Tensor(t->select(0, index));
  return torchffi_tensor_scalar(&tensor);
}

tensor torchffi_tensor_get(tensor t, int index) {
  auto tensor = t->select(0, index);
  return new torch::Tensor(tensor);
}

int8_t torchffi_tensor_get_datatype(tensor t) {
  return static_cast<int>(t->scalar_type());
}

tensor torchffi_tensor_to(tensor t, TensorOptions options, bool nonBlocking,
                          bool copy) {
  auto tensor = t->to(torchffi_make_tensor_options(options), nonBlocking, copy);
  return new torch::Tensor(tensor);
}

tensor torchffi_tensor_index(tensor t, Index_t *indices, size_t ndims) {
  std::vector<at::indexing::TensorIndex> indexer;
  for (int i = 0; i < ndims; i++) {
    indexer.push_back(torchffi_make_tensor_index(indices[i]));
  }
  at::Tensor tensor = t->index(at::ArrayRef(indexer.data(), ndims));
  return new torch::Tensor(tensor);
}

tensor torchffi_tensor_view(tensor t, int64_t *sizes, size_t ndims) {
  at::Tensor tensor = t->view(at::IntArrayRef(sizes, ndims));
  return new torch::Tensor(tensor);
}

tensor torchffi_tensor_reshape(tensor t, int64_t *sizes, size_t ndims) {
  at::Tensor tensor = t->reshape(at::IntArrayRef(sizes, ndims));
  return new torch::Tensor(tensor);
}

tensor torchffi_tensor_flatten(tensor t, int64_t startDim, int64_t endDim) {
  at::Tensor tensor = t->flatten(startDim, endDim);
  return new torch::Tensor(tensor);
}

tensor *torchffi_tensor_split_equally(tensor t, int64_t splits, int64_t dim) {
  auto tensors = t->split(splits, dim);
  tensor *result = (tensor *)malloc((tensors.size() + 1) * sizeof(tensor));
  for (int i = 0; i < tensors.size(); i++) {
    result[i] = new torch::Tensor(tensors[i]);
  }
  result[tensors.size()] = nullptr;

  return result;
}

tensor *torchffi_tensor_split(tensor t, int64_t *splits, size_t splitsSize,
                              int64_t dim) {
  auto tensors = t->split(at::IntArrayRef(splits, splitsSize), dim);
  tensor *result = (tensor *)malloc((tensors.size() + 1) * sizeof(tensor));
  for (int i = 0; i < tensors.size(); i++) {
    result[i] = new torch::Tensor(tensors[i]);
  }
  result[tensors.size()] = nullptr;

  return result;
}

tensor *torchffi_tensor_chunk(tensor t, int64_t chunks, int64_t dim) {
  auto tensors = t->chunk(chunks, dim);
  tensor *result = (tensor *)malloc((tensors.size() + 1) * sizeof(tensor));
  for (int i = 0; i < tensors.size(); i++) {
    result[i] = new torch::Tensor(tensors[i]);
  }
  result[tensors.size()] = nullptr;
  return result;
}

tensor torchffi_tensor_expand(tensor t, int64_t *sizes, size_t ndims,
                              bool implicit) {
  at::Tensor tensor = t->expand(at::IntArrayRef(sizes, ndims), implicit);
  return new torch::Tensor(tensor);
}

tensor torchffi_tensor_permute(tensor t, int64_t *dims, size_t ndims) {
  at::Tensor tensor = t->permute(at::IntArrayRef(dims, ndims));
  return new torch::Tensor(tensor);
}

tensor torchffi_tensor_transpose(tensor t, int64_t dim1, int64_t dim2) {
  at::Tensor tensor = t->transpose(dim1, dim2);
  return new torch::Tensor(tensor);
}

tensor torchffi_tensor_contiguous(tensor t, int8_t memoryFormat) {
  at::Tensor tensor = t->contiguous(at::MemoryFormat(memoryFormat));
  return new torch::Tensor(tensor);
}

tensor torchffi_tensor_squeeze(tensor t, int64_t *dim) {
  at::Tensor tensor;
  if (dim == nullptr) {
    tensor = t->squeeze();
  } else {
    tensor = t->squeeze(*dim);
  }
  return new torch::Tensor(tensor);
}

tensor torchffi_tensor_unsqueeze(tensor t, int64_t dim) {
  at::Tensor tensor = t->unsqueeze(dim);
  return new torch::Tensor(tensor);
}

const char *padModeName(uint8_t padMode) {
  switch (padMode) {
  case padModeConstant:
    return padModeNameConstant;
  case padModeReflect:
    return padModeNameReflect;
  case padModeReplicate:
    return padModeNameReplicate;
  case padModeCircular:
    return padModeNameCircular;
  default:
    return padModeNameConstant;
  }
}

tensor torchffi_tensor_pad(tensor t, int64_t *pad, size_t padArrayLength,
                           uint8_t padMode, double *value) {
  at::Tensor tensor =
      torch::pad(*t, at::IntArrayRef(pad, padArrayLength), padModeName(padMode),
                 value ? std::optional<double>(*value) : std::nullopt);
  return new torch::Tensor(tensor);
}

void torchffi_tensor_ones_(tensor t) { torch::nn::init::ones_(*t); }

void torchffi_tensor_zeros_(tensor t) { torch::nn::init::zeros_(*t); }

void torchffi_tensor_eye_(tensor t) { torch::nn::init::eye_(*t); }

void torchffi_tensor_fill_(tensor t, Scalar value) {
  at::Scalar opValue = torchffi_to_scalar(value);
  t->fill_(opValue);
}

void torchffi_tensor_rand_(tensor t, Generator generator) {
  std::optional<at::Generator> opGenerator = std::nullopt;
  if (generator != nullptr) {
    opGenerator = *generator;
  }
  t->random_(opGenerator);
}

void torchffi_tensor_normal_(tensor t, Generator generator, double mean,
                             double std) {
  std::optional<at::Generator> opGenerator = std::nullopt;
  if (generator != nullptr) {
    opGenerator = *generator;
  }
  t->normal_(mean, std, opGenerator);
}

void torchffi_tensor_uniform_(tensor t, Generator generator, double from,
                              double to) {
  std::optional<at::Generator> opGenerator = std::nullopt;
  if (generator != nullptr) {
    opGenerator = *generator;
  }
  t->uniform_(from, to, opGenerator);
}

bool torchffi_tensor_allclose(tensor a, tensor b, double rtol, double atol,
                              bool equalNan) {
  return a->allclose(*b, rtol, atol, equalNan);
}

tensor torchffi_tensor_addition(tensor a, tensor b, Scalar alpha) {
  at::Scalar opAlpha = torchffi_to_scalar(alpha);
  at::Tensor tensor = a->add(*b, opAlpha);
  return new torch::Tensor(tensor);
}

tensor torchffi_tensor_subtraction(tensor a, tensor b, Scalar alpha) {
  at::Scalar opAlpha = at::Scalar(1);
  if (alpha.dtype == 0) {
    opAlpha = at::Scalar(alpha.value.b);
  } else if (alpha.dtype == 1) {
    opAlpha = at::Scalar(alpha.value.i);
  } else if (alpha.dtype == 2) {
    opAlpha = at::Scalar(alpha.value.d);
  }
  at::Tensor tensor = a->sub(*b, opAlpha);
  return new torch::Tensor(tensor);
}

tensor torchffi_tensor_multiplication(tensor a, tensor b) {
  at::Tensor tensor = a->mul(*b);
  return new torch::Tensor(tensor);
}

tensor torchffi_tensor_division(tensor a, tensor b) {
  at::Tensor tensor = a->div(*b);
  return new torch::Tensor(tensor);
}

tensor torchffi_tensor_division_scalar(tensor a, Scalar b) {
  at::Scalar opB = torchffi_to_scalar(b);
  at::Tensor tensor = a->div(opB);
  return new torch::Tensor(tensor);
}

tensor torchffi_tensor_pow(tensor input, Scalar exponent) {
  at::Scalar opExponent = torchffi_to_scalar(exponent);
  at::Tensor tensor = torch::pow(*input, opExponent);
  return new torch::Tensor(tensor);
}

tensor torchffi_tensor_rsqrt(tensor input) {
  at::Tensor tensor = torch::rsqrt(*input);
  return new torch::Tensor(tensor);
}

tensor torchffi_tensor_sin(tensor input) {
  at::Tensor tensor = torch::sin(*input);
  return new torch::Tensor(tensor);
}

tensor torchffi_tensor_cos(tensor input) {
  at::Tensor tensor = torch::cos(*input);
  return new torch::Tensor(tensor);
}

tensor torchffi_tensor_exp(tensor input) {
  at::Tensor tensor = torch::exp(*input);
  return new torch::Tensor(tensor);
}

tensor torchffi_tensor_bitwise_not(tensor a) {
  at::Tensor tensor = a->bitwise_not();
  return new torch::Tensor(tensor);
}

tensor torchffi_tensor_bitwise_or(tensor a, tensor b) {
  at::Tensor tensor = a->bitwise_or(*b);
  return new torch::Tensor(tensor);
}

tensor torchffi_tensor_bitwise_and(tensor a, tensor b) {
  at::Tensor tensor = a->bitwise_and(*b);
  return new torch::Tensor(tensor);
}

tensor torchffi_tensor_bitwise_xor(tensor a, tensor b) {
  at::Tensor tensor = a->bitwise_xor(*b);
  return new torch::Tensor(tensor);
}

tensor torchffi_tensor_sum(tensor input, int64_t *dim, size_t dimLength,
                           bool keepdim, uint8_t *dtype) {
  std::optional<at::ScalarType> dopt = std::nullopt;
  if (dtype != nullptr) {
    dopt = at::ScalarType(*dtype);
  }
  at::OptionalIntArrayRef opDim = std::nullopt;
  if (dim != nullptr) {
    opDim = at::IntArrayRef(dim, dimLength);
  }
  at::Tensor tensor = torch::sum(*input, opDim, keepdim, dopt);
  return new torch::Tensor(tensor);
}

tensor torchffi_tensor_mean(tensor input, int64_t *dim, size_t dimLength,
                            bool keepdim, uint8_t *dtype) {
  std::optional<at::ScalarType> dopt = std::nullopt;
  if (dtype != nullptr) {
    dopt = at::ScalarType(*dtype);
  }
  at::OptionalIntArrayRef opDim = std::nullopt;
  if (dim != nullptr) {
    opDim = at::IntArrayRef(dim, dimLength);
  }
  at::Tensor tensor = torch::mean(*input, opDim, keepdim, dopt);
  return new torch::Tensor(tensor);
}

tensor torchffi_tensor_matmul(tensor a, tensor b) {
  at::Tensor tensor = a->matmul(*b);
  return new torch::Tensor(tensor);
}

tensor torchffi_tensor_sigmoid(tensor t) {
  return new torch::Tensor(t->sigmoid());
}

tensor torchffi_tensor_relu(tensor t) { return new torch::Tensor(t->relu()); }

tensor torchffi_tensor_gelu(tensor t, char *approximate) {
  return new torch::Tensor(torch::gelu(*t, approximate));
}

tensor torchffi_tensor_silu(tensor t) {
  return new torch::Tensor(torch::silu(*t));
}

tensor torchffi_linear(tensor input, tensor weight, tensor bias) {
  return new torch::Tensor(torch::linear(
      *input, *weight,
      (bias ? ::std::optional<at::Tensor>(*bias) : ::std::nullopt)));
}

tensor torchffi_layer_norm(tensor input, int64_t *normalizedShape,
                           size_t normalizedShapeLength, tensor weight,
                           tensor bias, double eps, bool cudnnEnable) {
  auto tensor = torch::layer_norm(
      *input, at::IntArrayRef(normalizedShape, normalizedShapeLength),
      (weight ? ::std::optional<at::Tensor>(*weight) : ::std::nullopt),
      (bias ? ::std::optional<at::Tensor>(*bias) : ::std::nullopt), eps,
      cudnnEnable);
  return new torch::Tensor(tensor);
}

tensor torchffi_group_norm(tensor input, int64_t numGroups, tensor weight,
                           tensor bias, float eps) {
  auto tensor = torch::group_norm(
      *input, numGroups,
      (weight ? ::std::optional<at::Tensor>(*weight) : ::std::nullopt),
      (bias ? ::std::optional<at::Tensor>(*bias) : ::std::nullopt), eps, true);
  return new torch::Tensor(tensor);
}

tensor torchffi_rms_norm(tensor input, int64_t *normalizedShape,
                         size_t normalizedShapeLength, tensor weight,
                         double *eps) {
  auto tensor = torch::rms_norm(
      *input, at::IntArrayRef(normalizedShape, normalizedShapeLength),
      (weight ? ::std::optional<at::Tensor>(*weight) : ::std::nullopt),
      (eps ? ::std::optional<double>(*eps) : ::std::nullopt));
  return new torch::Tensor(tensor);
}

tensor torchffi_dropout(tensor t, double p, bool train) {
  at::Tensor tensor = torch::dropout(*t, p, train);
  return new torch::Tensor(tensor);
}

void torchffi_dropout_(tensor t, double p, bool train) {
  torch::dropout_(*t, p, train);
}

tensor torchffi_tensor_softmax(tensor t, int64_t dim, uint8_t *dataType) {
  std::optional<at::ScalarType> dtype = std::nullopt;
  if (dataType != nullptr) {
    dtype = at::ScalarType(*dataType);
  }
  at::Tensor tensor = torch::softmax(*t, dim, dtype);
  return new torch::Tensor(tensor);
}

tensor torchffi_embedding_renorm_(tensor weights, tensor indices,
                                  double maxNorm, double normType) {
  at::Tensor tensor =
      torch::embedding_renorm_(*weights, *indices, maxNorm, normType);
  return new torch::Tensor(tensor);
}

tensor torchffi_embedding(tensor weights, tensor indices, int64_t paddingIdx,
                          bool scaleGradByFreq, bool sparse) {
  at::Tensor tensor =
      torch::embedding(*weights, *indices, paddingIdx, scaleGradByFreq, sparse);
  return new torch::Tensor(tensor);
}

tensor torchffi_conv2d(tensor input, tensor weights, tensor bias,
                       int64_t *strides, int64_t *paddings, int64_t *dilations,
                       int64_t groups) {
  at::Tensor tensor =
      torch::conv2d(*input, *weights,
                    (bias ? std::optional<at::Tensor>(*bias) : ::std::nullopt),
                    at::IntArrayRef(strides, 2), at::IntArrayRef(paddings, 2),
                    at::IntArrayRef(dilations, 2), groups);
  return new torch::Tensor(tensor);
}

tensor torchffi_upsample_nearest(tensor input, int64_t *outputSize,
                                 size_t outputSizeLength) {
  at::Tensor tensor;
  switch (outputSizeLength) {
  case 1:
    tensor = torch::upsample_nearest1d(
        *input,
        std::optional<at::IntArrayRef>(
            at::IntArrayRef(outputSize, outputSizeLength)),
        ::std::nullopt);
    return new torch::Tensor(tensor);
  case 2:
    tensor = torch::upsample_nearest2d(
        *input,
        std::optional<at::IntArrayRef>(
            at::IntArrayRef(outputSize, outputSizeLength)),
        ::std::nullopt);
    return new torch::Tensor(tensor);
  case 3:
    tensor = torch::upsample_nearest3d(
        *input,
        std::optional<at::IntArrayRef>(
            at::IntArrayRef(outputSize, outputSizeLength)),
        ::std::nullopt);
    return new torch::Tensor(tensor);
  default:
    return nullptr;
  }
}

tensor torchffi_upsample_nearest_scale(tensor input, double *scales,
                                       size_t scalesLength) {
  at::Tensor tensor;
  switch (scalesLength) {
  case 1:
    tensor = torch::upsample_nearest1d(
        *input, ::std::nullopt,
        std::optional<at::ArrayRef<double>>(
            at::ArrayRef<double>(scales, scalesLength)));
    return new torch::Tensor(tensor);
  case 2:
    tensor = torch::upsample_nearest2d(
        *input, ::std::nullopt,
        std::optional<at::ArrayRef<double>>(
            at::ArrayRef<double>(scales, scalesLength)));
    return new torch::Tensor(tensor);
  case 3:
    tensor = torch::upsample_nearest3d(
        *input, ::std::nullopt,
        std::optional<at::ArrayRef<double>>(
            at::ArrayRef<double>(scales, scalesLength)));
    return new torch::Tensor(tensor);
  default:
    return nullptr;
  }
}

tensor torchffi_upsample_nearest_exact(tensor input, int64_t *outputSize,
                                       size_t outputSizeLength) {
  at::Tensor tensor;
  switch (outputSizeLength) {
  case 1:
    tensor = torch::_upsample_nearest_exact1d(
        *input,
        std::optional<at::IntArrayRef>(
            at::IntArrayRef(outputSize, outputSizeLength)),
        ::std::nullopt);
    return new torch::Tensor(tensor);
  case 2:
    tensor = torch::_upsample_nearest_exact2d(
        *input,
        std::optional<at::IntArrayRef>(
            at::IntArrayRef(outputSize, outputSizeLength)),
        ::std::nullopt);
    return new torch::Tensor(tensor);
  case 3:
    tensor = torch::_upsample_nearest_exact3d(
        *input,
        std::optional<at::IntArrayRef>(
            at::IntArrayRef(outputSize, outputSizeLength)),
        ::std::nullopt);
    return new torch::Tensor(tensor);
  default:
    return nullptr;
  }
}

tensor torchffi_upsample_nearest_exact_scale(tensor input, double *scales,
                                             size_t scalesLength) {
  at::Tensor tensor;
  switch (scalesLength) {
  case 1:
    tensor = torch::_upsample_nearest_exact1d(
        *input, ::std::nullopt,
        std::optional<at::ArrayRef<double>>(
            at::ArrayRef<double>(scales, scalesLength)));
    return new torch::Tensor(tensor);
  case 2:
    tensor = torch::_upsample_nearest_exact1d(
        *input, ::std::nullopt,
        std::optional<at::ArrayRef<double>>(
            at::ArrayRef<double>(scales, scalesLength)));
    return new torch::Tensor(tensor);
  case 3:
    tensor = torch::_upsample_nearest_exact1d(
        *input, ::std::nullopt,
        std::optional<at::ArrayRef<double>>(
            at::ArrayRef<double>(scales, scalesLength)));
    return new torch::Tensor(tensor);
  default:
    return nullptr;
  }
}

tensor torchffi_avg_pool2d(tensor input, int64_t kernelSizeH,
                           int64_t kernelSizeW, int64_t strideH,
                           int64_t strideW, int64_t paddingH, int64_t paddingW,
                           bool ceilMode, bool countIncludePad,
                           int64_t *divisorOverride) {
  at::Tensor tensor = torch::avg_pool2d(
      *input, at::IntArrayRef({kernelSizeH, kernelSizeW}),
      at::IntArrayRef({strideH, strideW}),
      at::IntArrayRef({paddingH, paddingW}), ceilMode, countIncludePad,
      divisorOverride ? std::optional<int64_t>(*divisorOverride)
                      : std::nullopt);
  return new torch::Tensor(tensor);
}