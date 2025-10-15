import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart' as ffi;

final ffi.DynamicLibrary nativeLib = ffi.DynamicLibrary.open(
  'torchffi/src/build/libtorchffi.dylib',
);

extension type Tensor(ffi.Pointer<ffi.Void> _tensor) {
  static Tensor eye(int n) {
    final ffi.Pointer<ffi.Pointer<ffi.Void>> tensor = ffi.malloc
        .allocate<ffi.Pointer<ffi.Void>>(ffi.sizeOf<ffi.Pointer<ffi.Void>>());
    try {
      TensorFFI.eye(tensor, n);
      return Tensor(tensor.value);
    } finally {
      ffi.malloc.free(tensor);
    }
  }
}

abstract class TensorFFI {
  static final void Function(ffi.Pointer<ffi.Pointer<ffi.Void>> tensor, int n)
  eye = nativeLib
      .lookupFunction<
        ffi.Void Function(ffi.Pointer<ffi.Pointer<ffi.Void>>, ffi.Int64),
        void Function(ffi.Pointer<ffi.Pointer<ffi.Void>> tensor, int n)
      >('torchffi_new_tensor_eye');
}

/*
class Tensor {
  
  external ffi.Pointer<ffi.Void> _createTensor();
  external void _freeTensor(ffi.Pointer<ffi.Void> tensor);

  late final ffi.Pointer<ffi.Void> _tensor;
  Tensor() {
    _tensor = _createTensor();
  }

  /// Frees the underlying C++ tensor.
  void dispose() {
    _freeTensor(_tensor);
  }
}*/

final class TensorOptions extends ffi.Struct {
  @ffi.Uint8()
  external int dataType;
  @ffi.Int8()
  external int device;
  // TODO memory layout
}
