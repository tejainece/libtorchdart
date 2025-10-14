// TODO: Put public facing types in this file.

import 'dart:ffi';

final DynamicLibrary nativeLib = DynamicLibrary.open('libtorch_linux/lib/libtorch_cuda.so');

class Tensor {
  external Pointer<Void> _createTensor();
  external void _freeTensor(Pointer<Void> tensor);

  late final Pointer<Void> _tensor;
  Tensor() {
    _tensor = _createTensor();
  }

  /// Frees the underlying C++ tensor.
  void dispose() {
    _freeTensor(_tensor);
  }
}