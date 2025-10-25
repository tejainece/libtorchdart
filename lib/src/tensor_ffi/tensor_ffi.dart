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

  void setBool(bool value) {
    dataType = 0;
    _value.b = value;
  }

  void setInt(int value) {
    dataType = 1;
    _value.i = value;
  }

  void setDouble(double value) {
    dataType = 2;
    _value.d = value;
  }

  void setValue(dynamic value) {
    if (value is bool) {
      setBool(value);
    } else if (value is int) {
      setInt(value);
    } else if (value is double) {
      setDouble(value);
    } else {
      throw ArgumentError('Unsupported scalar type: ${value.runtimeType}');
    }
  }

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

  static Pointer<FFIScalar> allocate(Allocator allocator) =>
      malloc.allocate<FFIScalar>(sizeOf<FFIScalar>());
}

typedef CTensor = Pointer<Void>;

abstract class TensorFFI {
  static final constructor = nativeLib
      .lookupFunction<CTensor Function(), CTensor Function()>(
        'torchffi_tensor_new',
      );

  static final zeros = nativeLib
      .lookupFunction<
        CTensor Function(Pointer<Int64>, Size, FFITensorOptions),
        CTensor Function(Pointer<Int64>, int dims, FFITensorOptions)
      >('torchffi_tensor_new_zeros');

  static final ones = nativeLib
      .lookupFunction<
        CTensor Function(Pointer<Int64>, Size, FFITensorOptions),
        CTensor Function(Pointer<Int64>, int dims, FFITensorOptions)
      >('torchffi_tensor_new_ones');

  static final arange = nativeLib
      .lookupFunction<
        CTensor Function(Int64 end, FFITensorOptions),
        CTensor Function(int end, FFITensorOptions)
      >('torchffi_tensor_new_arange');

  static final rand = nativeLib
      .lookupFunction<
        CTensor Function(Pointer<Int64>, Size, FFITensorOptions),
        CTensor Function(Pointer<Int64>, int dims, FFITensorOptions)
      >('torchffi_tensor_new_rand');

  static final eye = nativeLib
      .lookupFunction<
        CTensor Function(Int64, Int64, FFITensorOptions),
        CTensor Function(int n, int m, FFITensorOptions options)
      >('torchffi_tensor_new_eye');

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

  static final expand = nativeLib
      .lookupFunction<
        CTensor Function(CTensor, Pointer<Int64>, Size, Bool),
        CTensor Function(CTensor, Pointer<Int64>, int, bool)
      >('torchffi_tensor_expand');

  static final addition = nativeLib
      .lookupFunction<
        CTensor Function(CTensor, CTensor, FFIScalar alpha),
        CTensor Function(CTensor tensor1, CTensor tensor2, FFIScalar alpha)
      >('torchffi_tensor_addition');

  static final subtraction = nativeLib
      .lookupFunction<
        CTensor Function(CTensor, CTensor, FFIScalar alpha),
        CTensor Function(CTensor tensor1, CTensor tensor2, FFIScalar alpha)
      >('torchffi_tensor_subtraction');

  static final multiplication = nativeLib
      .lookupFunction<
        CTensor Function(CTensor, CTensor),
        CTensor Function(CTensor tensor1, CTensor tensor2)
      >('torchffi_tensor_multiplication');

  static final division = nativeLib
      .lookupFunction<
        CTensor Function(CTensor, CTensor),
        CTensor Function(CTensor tensor1, CTensor tensor2)
      >('torchffi_tensor_division');

  static final sigmoid = nativeLib
      .lookupFunction<
        CTensor Function(CTensor),
        CTensor Function(CTensor tensor)
      >('torchffi_tensor_sigmoid');

  static final gelu = nativeLib
      .lookupFunction<
        CTensor Function(CTensor, Pointer<Utf8>),
        CTensor Function(CTensor tensor, Pointer<Utf8>)
      >('torchffi_tensor_gelu');

  static final embedding = nativeLib
      .lookupFunction<
        CTensor Function(CTensor, CTensor, Int64, Bool, Bool),
        CTensor Function(
          CTensor tensor,
          CTensor weights,
          int paddingIdx,
          bool scaleGradByFreq,
          bool sparse,
        )
      >('torchffi_embedding');
}
