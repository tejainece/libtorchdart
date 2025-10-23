#include <torch/all.h>
#include <torch_ffi.h>

tensor torchffi_new_tensor_eye(int64_t n, int64_t m, TensorOptions options) {
    at::TensorOptions tensorOptions = at::device(at::Device(at::DeviceType(options.deviceType), options.deviceIndex))
        .dtype(at::ScalarType(options.dtype))
        .layout(at::Layout(options.layout));
    // TODO memory layout
    // TODO autograd
    // TODO pinned memory

    at::Tensor tensor = torch::eye(n, m, tensorOptions);
    return new torch::Tensor(tensor);
}

tensor torchffi_tensor_new_from_blob(void *data, int64_t *dims, size_t ndims, TensorOptions options) {
    at::TensorOptions tensorOptions = at::device(at::Device(at::DeviceType(options.deviceType), options.deviceIndex))
        .dtype(at::ScalarType(options.dtype))
        .layout(at::Layout(options.layout));
    return new torch::Tensor(torch::for_blob(data, torch::IntArrayRef(dims, ndims)).options(tensorOptions).make_tensor());
}

/*tensor torchffi_tensor_new_from_array(void *data, int64_t *dims, size_t ndims, int64_t *strides, size_t nstrides, int type, int device) {
    at::TensorOptions blobOptions = at::TensorOptions()
        //.device(device_of_int(device))
        .dtype(torch::ScalarType(type));
    if (nstrides == 0) {
      return new torch::Tensor(torch::for_blob(data, torch::IntArrayRef(dims, ndims)).options(blobOptions).make_tensor());
    } else {
      return new torch::Tensor(torch::from_blob(data, torch::IntArrayRef(dims, ndims), torch::IntArrayRef(strides, nstrides), blobOptions));
    }
}

tensor torchffi_tensor_of_data(void *vs, int64_t *dims, size_t ndims, size_t element_size_in_bytes, int type) {
    torch::Tensor tensor = torch::zeros(torch::IntArrayRef(dims, ndims), torch::ScalarType(type));
    if ((int64_t)element_size_in_bytes != tensor.element_size())
        throw std::invalid_argument("incoherent element sizes in bytes");
    void *tensor_data = tensor.data_ptr();
    memcpy(tensor_data, vs, tensor.numel() * element_size_in_bytes);
    return new torch::Tensor(tensor);
}*/

tensor  torchffi_new_tensor() {
    return new torch::Tensor();
}

size_t  torchffi_tensor_dim(tensor t) {
  return t->dim();
}

void  torchffi_tensor_sizes(tensor t, size_t dim, int64_t *shape) {
    int i = 0;
    for (int64_t dimShape : t->sizes()) {
        if(i == dim) {
            break;
        }
        shape[i++] = dimShape;
    }
    for(; i < dim; i++) {
        shape[i] = 0;
    }
}

Device torchffi_tensor_device(tensor t) {
    auto device = t->device();
    return Device{int8_t(device.type()), device.index()};
}

Scalar_t torchffi_tensor_item(tensor t) {
    at::Scalar scalar = t->item();
    at::ScalarType type = scalar.type();
    if(scalar.isBoolean()) {
        return {
            .dtype = 0,
            .value = {
                .b = scalar.toBool()
            },
        };
    } else if(scalar.isIntegral(false)) {
        return {
            .dtype = 1,
            .value = {
                .i = scalar.toLong()
            },
        };
    } else if(scalar.isFloatingPoint()) {
        return {
            .dtype = 2,
            .value = {
                .d = scalar.toDouble()
            },
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

tensor torchffi_embedding(tensor weights, tensor indices, int64_t paddingIdx, bool scaleGradByFreq, bool sparse) {
    at::Tensor tensor = torch::embedding(*weights, *indices, paddingIdx, scaleGradByFreq, sparse);
    return new torch::Tensor(tensor);
}

/*
void _scratch(tensor t) {
    t->view;
    t->view_as;
}*/