import 'dart:ffi';
import 'package:ffi/ffi.dart';

import 'package:libtorchdart/libtorchdart.dart';

final DynamicLibrary nativeLib = DynamicLibrary.open(
  'torchffi/src/build/libtorchffi.dylib',
);

final class FFIDevice extends Struct {
  @Int8()
  external int deviceType;
  @Int8()
  external int deviceIndex;
}

final class FFITensorOptions extends Struct {
  @Int8()
  external int dataType;
  @Int8()
  external int deviceType;
  @Int8()
  external int deviceIndex;
  @Int8()
  external int layout;

  static Pointer<FFITensorOptions> allocate(Allocator allocator) =>
      malloc.allocate<FFITensorOptions>(sizeOf<FFITensorOptions>());

  static Pointer<FFITensorOptions> make({
    required DataType dataType,
    required Device device,
    required Layout layout,
    required Allocator allocator,
  }) {
    final options = allocate(allocator);
    options.ref.dataType = dataType.type;
    options.ref.deviceType = device.deviceType.type;
    options.ref.deviceIndex = device.deviceIndex;
    options.ref.layout = layout.type;
    return options;
  }
}

final class _ScalarValue extends Union {
  @Bool()
  external bool b;

  @Int64()
  external int i;

  @Double()
  external double d;
}

final class FFIScalar extends Struct {
  @Int8()
  external int dataType;

  external _ScalarValue _value;

  dynamic get value {
    switch (dataType) {
      case 0:
        return _value.b;
      case 1:
        return _value.i;
      case 2:
        return _value.d;
      default:
        return null;
    }
  }
}

typedef CTensor = Pointer<Void>;

abstract class TensorFFI {
  static final dim = nativeLib
      .lookupFunction<Size Function(CTensor), int Function(CTensor tensor)>(
        'torchffi_tensor_dim',
      );

  static final sizes = nativeLib
      .lookupFunction<
        Void Function(CTensor, Int64, Pointer<Int64>),
        void Function(CTensor, int dim, Pointer<Int64>)
      >('torchffi_tensor_sizes');

  static final tensorGetDevice = nativeLib
      .lookupFunction<
        FFIDevice Function(CTensor),
        FFIDevice Function(CTensor tensor)
      >('torchffi_tensor_device');

  static final eye = nativeLib
      .lookupFunction<
        CTensor Function(Int64, Int64, FFITensorOptions),
        CTensor Function(int n, int m, FFITensorOptions options)
      >('torchffi_new_tensor_eye');

  static final fromBlob = nativeLib
      .lookupFunction<
        CTensor Function(
          Pointer<Void>,
          Pointer<Int64>,
          Size dims,
          FFITensorOptions,
        ),
        CTensor Function(
          Pointer<Void>,
          Pointer<Int64>,
          int dims,
          FFITensorOptions,
        )
      >('torchffi_tensor_new_from_blob');

  static final item = nativeLib
      .lookupFunction<
        FFIScalar Function(CTensor),
        FFIScalar Function(CTensor tensor)
      >('torchffi_tensor_item');

  static final get = nativeLib
      .lookupFunction<
        CTensor Function(CTensor, Int64),
        CTensor Function(CTensor tensor, int index)
      >('torchffi_tensor_get');
}
