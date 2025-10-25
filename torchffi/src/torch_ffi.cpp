#include <torch/all.h>
#include <torch_ffi.h>

at::TensorOptions torchffi_make_tensor_options(TensorOptions options) {
    return at::device(at::Device(at::DeviceType(options.deviceType), options.deviceIndex))
        .dtype(at::ScalarType(options.dtype))
        .layout(at::Layout(options.layout));
    // TODO memory layout
    // TODO autograd
    // TODO pinned memory
}

tensor torchffi_new_tensor() {
    return new torch::Tensor();
}

tensor torchffi_tensor_new_zeros(int64_t *sizes, size_t ndims, TensorOptions options) {
    at::Tensor tensor = torch::zeros(at::IntArrayRef(sizes, ndims), torchffi_make_tensor_options(options));
    return new torch::Tensor(tensor);
}

tensor torchffi_tensor_new_ones(int64_t *sizes, size_t ndims, TensorOptions options) {
    at::Tensor tensor = torch::ones(at::IntArrayRef(sizes, ndims), torchffi_make_tensor_options(options));
    return new torch::Tensor(tensor);
}

tensor torchffi_tensor_new_arange(int64_t end, TensorOptions options) {
    at::Tensor tensor = torch::arange(end, torchffi_make_tensor_options(options));
    return new torch::Tensor(tensor);
}

tensor torchffi_tensor_new_rand(int64_t *sizes, size_t ndims, TensorOptions options) {
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

Scalar torchffi_tensor_item(tensor t) {
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

tensor torchffi_tensor_get(tensor t, int index) {
    return new torch::Tensor((*t)[index]);
}

tensor torchffi_tensor_expand(tensor t, int64_t *sizes, size_t ndims, bool implicit) {
    at::Tensor tensor = t->expand(at::IntArrayRef(sizes, ndims), implicit);
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

tensor torchffi_tensor_sigmoid(tensor t) {
    return new torch::Tensor(t->sigmoid());
}

tensor torchffi_tensor_gelu(tensor t, char* approximate) {
    return new torch::Tensor(torch::gelu(*t, approximate));
}

tensor torchffi_embedding(tensor weights, tensor indices, int64_t paddingIdx, bool scaleGradByFreq, bool sparse) {
    at::Tensor tensor = torch::embedding(*weights, *indices, paddingIdx, scaleGradByFreq, sparse);
    return new torch::Tensor(tensor);
}