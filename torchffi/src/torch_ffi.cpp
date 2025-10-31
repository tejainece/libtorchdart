#include <torch/all.h>
#include <torch_ffi.h>

#include <optional>

at::TensorOptions torchffi_make_tensor_options(TensorOptions options) {
    return at::device(at::Device(at::DeviceType(options.deviceType), options.deviceIndex))
        .dtype(at::ScalarType(options.dtype))
        .layout(at::Layout(options.layout))
        .memory_format(at::MemoryFormat(options.memoryFormat));
    // TODO memory layout
    // TODO autograd
    // TODO pinned memory
}

at::indexing::TensorIndex torchffi_make_tensor_index(Index_t& index) {
    at::indexing::TensorIndexType type = (at::indexing::TensorIndexType)index.type;
    if (type == at::indexing::TensorIndexType::None) {
        return at::indexing::None;
    } else if (type == at::indexing::TensorIndexType::Ellipsis) {
        return at::indexing::Ellipsis;
    } else if (type == at::indexing::TensorIndexType::SymInt) {
        return at::indexing::TensorIndex(*(int64_t*)index.value);
    } else if (type == at::indexing::TensorIndexType::Boolean) {
        return at::indexing::TensorIndex(*(bool*)index.value);
    } else if (type == at::indexing::TensorIndexType::Slice) {
        Slice slice = *(Slice*)index.value;
        return at::indexing::TensorIndex(at::indexing::Slice(
            slice.start ? std::make_optional(c10::SymInt(*slice.start)) : std::nullopt,
            slice.stop ? std::make_optional(c10::SymInt(*slice.stop)) : std::nullopt,
            c10::SymInt(slice.step)));
    } else if (type == at::indexing::TensorIndexType::Tensor) {
        return at::indexing::TensorIndex(*(torch::Tensor*)index.value);
    } else {
        return at::indexing::Ellipsis;
    }
}

void torchffi_tensor_delete(tensor t) {
    delete t;
}

tensor torchffi_tensor_new() {
    return new torch::Tensor();
}

tensor torchffi_tensor_new_zeros(int64_t* sizes, size_t ndims, TensorOptions options) {
    at::Tensor tensor = torch::zeros(at::IntArrayRef(sizes, ndims), torchffi_make_tensor_options(options));
    return new torch::Tensor(tensor);
}

tensor torchffi_tensor_new_ones(int64_t* sizes, size_t ndims, TensorOptions options) {
    at::Tensor tensor = torch::ones(at::IntArrayRef(sizes, ndims), torchffi_make_tensor_options(options));
    return new torch::Tensor(tensor);
}

tensor torchffi_tensor_new_arange(int64_t end, TensorOptions options) {
    at::Tensor tensor = torch::arange(end, torchffi_make_tensor_options(options));
    return new torch::Tensor(tensor);
}

tensor torchffi_tensor_new_rand(int64_t* sizes, size_t ndims, TensorOptions options) {
    at::Tensor tensor = torch::rand(at::IntArrayRef(sizes, ndims), torchffi_make_tensor_options(options));
    return new torch::Tensor(tensor);
}

tensor torchffi_tensor_new_eye(int64_t n, int64_t m, TensorOptions options) {
    at::Tensor tensor = torch::eye(n, m, torchffi_make_tensor_options(options));
    return new torch::Tensor(tensor);
}

tensor torchffi_tensor_new_from_blob(void* data, int64_t* dims, size_t ndims, TensorOptions options) {
    at::TensorOptions tensorOptions = torchffi_make_tensor_options(options);
    return new torch::Tensor(torch::for_blob(data, torch::IntArrayRef(dims, ndims)).options(tensorOptions).make_tensor());
}

size_t torchffi_tensor_dim(tensor t) {
    return t->dim();
}

void torchffi_tensor_sizes(tensor t, size_t dim, int64_t* shape) {
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
            .value = {
                .b = scalar.toBool()},
        };
    } else if (scalar.isIntegral(false)) {
        return {
            .dtype = 1,
            .value = {
                .i = scalar.toLong()},
        };
    } else if (scalar.isFloatingPoint()) {
        return {
            .dtype = 2,
            .value = {
                .d = scalar.toDouble()},
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

tensor torchffi_tensor_index(tensor t, Index_t* indices, size_t ndims) {
    std::vector<at::indexing::TensorIndex> indexer;
    for (int i = 0; i < ndims; i++) {
        indexer.push_back(torchffi_make_tensor_index(indices[i]));
    }
    at::Tensor tensor = t->index(at::ArrayRef(indexer.data(), ndims));
    return new torch::Tensor(tensor);
}

tensor torchffi_tensor_view(tensor t, int64_t* sizes, size_t ndims) {
    at::Tensor tensor = t->view(at::IntArrayRef(sizes, ndims));
    return new torch::Tensor(tensor);
}

tensor torchffi_tensor_expand(tensor t, int64_t* sizes, size_t ndims, bool implicit) {
    at::Tensor tensor = t->expand(at::IntArrayRef(sizes, ndims), implicit);
    return new torch::Tensor(tensor);
}

tensor torchffi_tensor_permute(tensor t, int64_t* dims, size_t ndims) {
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

const char* padModeName(uint8_t padMode) {
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

tensor torchffi_tensor_pad(tensor t, int64_t* pad, size_t padArrayLength, uint8_t padMode, double* value) {
    at::Tensor tensor = torch::pad(*t, at::IntArrayRef(pad, padArrayLength), padModeName(padMode), value ? std::optional<double>(*value) : std::nullopt);
    return new torch::Tensor(tensor);
}

tensor torchffi_tensor_addition(tensor a, tensor b, Scalar alpha) {
    at::Scalar opAlpha = at::Scalar(1);
    if (alpha.dtype == 0) {
        opAlpha = at::Scalar(alpha.value.b);
    } else if (alpha.dtype == 1) {
        opAlpha = at::Scalar(alpha.value.i);
    } else if (alpha.dtype == 2) {
        opAlpha = at::Scalar(alpha.value.d);
    }

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

tensor torchffi_tensor_matmul(tensor a, tensor b) {
    at::Tensor tensor = a->matmul(*b);
    return new torch::Tensor(tensor);
}

tensor torchffi_tensor_sigmoid(tensor t) {
    return new torch::Tensor(t->sigmoid());
}

tensor torchffi_tensor_gelu(tensor t, char* approximate) {
    return new torch::Tensor(torch::gelu(*t, approximate));
}

tensor torchffi_linear(tensor input, tensor weight, tensor bias) {
    return new torch::Tensor(torch::linear(*input, *weight, (bias ? ::std::optional<at::Tensor>(*bias) : ::std::nullopt)));
}

tensor torchffi_layer_norm(tensor input, int64_t* normalizedShape, size_t normalizedShapeLength, tensor weight, tensor bias, double eps, bool cudnnEnable) {
    return new torch::Tensor(torch::layer_norm(*input, at::IntArrayRef(normalizedShape, normalizedShapeLength),
                                               (weight ? ::std::optional<at::Tensor>(*weight) : ::std::nullopt),
                                               (bias ? ::std::optional<at::Tensor>(*bias) : ::std::nullopt),
                                               eps, cudnnEnable));
}

tensor torchffi_tensor_dropout(tensor t, double p, bool train) {
    at::Tensor tensor = torch::dropout(*t, p, train);
    return new torch::Tensor(tensor);
}

tensor torchffi_tensor_softmax(tensor t, int64_t dim, uint8_t* dataType) {
    std::optional<at::ScalarType> dtype = std::nullopt;
    if (dataType != nullptr) {
        dtype = at::ScalarType(*dataType);
    }
    at::Tensor tensor = torch::softmax(*t, dim, dtype);
    return new torch::Tensor(tensor);
}

tensor torchffi_embedding(tensor weights, tensor indices, int64_t paddingIdx, bool scaleGradByFreq, bool sparse) {
    at::Tensor tensor = torch::embedding(*weights, *indices, paddingIdx, scaleGradByFreq, sparse);
    return new torch::Tensor(tensor);
}

tensor torchffi_conv2d(tensor input, tensor weights, tensor bias, int64_t* strides, int64_t* paddings, int64_t* dilations, int64_t groups) {
    at::Tensor tensor = torch::conv2d(*input, *weights, (bias ? std::optional<at::Tensor>(*bias) : ::std::nullopt),
                                      at::IntArrayRef(strides, 2),
                                      at::IntArrayRef(paddings, 2),
                                      at::IntArrayRef(dilations, 2),
                                      groups);
    return new torch::Tensor(tensor);
}