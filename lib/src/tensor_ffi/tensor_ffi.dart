import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart' as ffi;

import 'package:libtorchdart/libtorchdart.dart';

final ffi.DynamicLibrary nativeLib = ffi.DynamicLibrary.open(
  'torchffi/src/build/libtorchffi.dylib',
);

final class FFIDevice extends ffi.Struct {
  @ffi.Int8()
  external int deviceType;
  @ffi.Int8()
  external int deviceIndex;
}

final class FFITensorOptions extends ffi.Struct {
  @ffi.Int8()
  external int dataType;
  @ffi.Int8()
  external int deviceType;
  @ffi.Int8()
  external int deviceIndex;
  @ffi.Int8()
  external int layout;

  static ffi.Pointer<FFITensorOptions> allocate() =>
      ffi.malloc.allocate<FFITensorOptions>(ffi.sizeOf<FFITensorOptions>());

  static ffi.Pointer<FFITensorOptions> make({
    required DataType dataType,
    required Device device,
    required Layout layout,
  }) {
    final options = allocate();
    options.ref.dataType = dataType.type;
    options.ref.deviceType = device.deviceType.type;
    options.ref.deviceIndex = device.deviceIndex;
    options.ref.layout = layout.type;
    return options;
  }
}

abstract class TensorFFI {
  static final dim = nativeLib
      .lookupFunction<
        ffi.Size Function(ffi.Pointer<ffi.Void>),
        int Function(ffi.Pointer<ffi.Void> tensor)
      >('torchffi_tensor_dim');

  static final sizes = nativeLib
      .lookupFunction<
        ffi.Void Function(
          ffi.Pointer<ffi.Void>,
          ffi.Int64,
          ffi.Pointer<ffi.Int64>,
        ),
        void Function(ffi.Pointer<ffi.Void>, int dim, ffi.Pointer<ffi.Int64>)
      >('torchffi_tensor_sizes');

  static final tensorGetDevice = nativeLib
      .lookupFunction<
        FFIDevice Function(ffi.Pointer<ffi.Void>),
        FFIDevice Function(ffi.Pointer<ffi.Void> tensor)
      >('torchffi_tensor_device');

  static final eye = nativeLib
      .lookupFunction<
        ffi.Pointer<ffi.Void> Function(ffi.Int64, ffi.Int64, FFITensorOptions),
        ffi.Pointer<ffi.Void> Function(int n, int m, FFITensorOptions options)
      >('torchffi_new_tensor_eye');
}
