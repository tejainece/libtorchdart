import 'dart:ffi';
import 'package:ffi/ffi.dart';

import 'package:libtorchdart/libtorchdart.dart';
import 'package:libtorchdart/src/nn/pooling.dart';

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
  @Int8()
  external int memoryFormat;

  static Pointer<FFITensorOptions> allocate(Allocator allocator) =>
      allocator.allocate<FFITensorOptions>(sizeOf<FFITensorOptions>());

  static Pointer<FFITensorOptions> make({
    required DataType dataType,
    required Device device,
    required Layout layout,
    required MemoryFormat memoryFormat,
    required Allocator allocator,
  }) {
    final options = allocate(allocator);
    options.ref.dataType = dataType.type;
    options.ref.deviceType = device.deviceType.type;
    options.ref.deviceIndex = device.deviceIndex;
    options.ref.layout = layout.type;
    options.ref.memoryFormat = memoryFormat.id;
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

final class FFISlice extends Struct {
  external Pointer<Int64> start;
  external Pointer<Int64> end;
  @Int64()
  external int step;

  static Pointer<FFISlice> fromSlice(Slice slice, Allocator allocator) {
    final ffiSlice = allocator.allocate<FFISlice>(sizeOf<FFISlice>());
    ffiSlice.ref.step = slice.step;
    if (slice.start != null) {
      ffiSlice.ref.start = allocator.allocate<Int64>(sizeOf<Int64>());
      ffiSlice.ref.start.value = slice.start!;
    } else {
      ffiSlice.ref.start = nullptr;
    }
    if (slice.end != null) {
      ffiSlice.ref.end = allocator.allocate<Int64>(sizeOf<Int64>());
      ffiSlice.ref.end.value = slice.end!;
    } else {
      ffiSlice.ref.end = nullptr;
    }
    return ffiSlice;
  }
}

final class FFIIndex extends Struct {
  @Int8()
  external int type;
  external Pointer<Void> value;

  void fromIndex(dynamic index, Allocator allocator) {
    if (index is int) {
      type = FFIIndexType.intType.index;
      value = (allocator.allocate<Int64>(
        sizeOf<Int64>(),
      )..value = index).cast();
    } else if (index is Slice) {
      type = FFIIndexType.sliceType.index;
      value = FFISlice.fromSlice(index, allocator).cast();
    } else if (index is Ellipsis) {
      type = FFIIndexType.ellipsisType.index;
      value = nullptr;
    } else if (index is NewDim) {
      type = FFIIndexType.newDimType.index;
      value = nullptr;
    } else if (index is Tensor) {
      type = FFIIndexType.tensorType.index;
      value = index.nativePtr;
    } else if (index is bool) {
      type = FFIIndexType.boolType.index;
      value = (allocator.allocate<Bool>(sizeOf<Bool>())..value = index).cast();
    } else {
      throw UnimplementedError('Unsupported index type: ${index.runtimeType}');
    }
  }
}

enum FFIIndexType {
  newDimType,
  ellipsisType,
  intType,
  boolType,
  sliceType,
  tensorType,
}

abstract class Torch {
  static final constructor = nativeLib
      .lookupFunction<CTensor Function(), CTensor Function()>(
        'torchffi_tensor_new',
      );

  static final delete = nativeLib
      .lookup<NativeFunction<Void Function(Pointer<Void>)>>(
        'torchffi_tensor_delete',
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

  static final scalar = nativeLib
      .lookupFunction<
        FFIScalar Function(CTensor),
        FFIScalar Function(CTensor tensor)
      >('torchffi_tensor_scalar');

  static final scalarAt = nativeLib
      .lookupFunction<
        FFIScalar Function(CTensor, Int64),
        FFIScalar Function(CTensor, int)
      >('torchffi_tensor_scalar_at');

  static final get = nativeLib
      .lookupFunction<
        CTensor Function(CTensor, Int64),
        CTensor Function(CTensor tensor, int index)
      >('torchffi_tensor_get');

  static final index = nativeLib
      .lookupFunction<
        CTensor Function(CTensor, Pointer<FFIIndex>, Int64),
        CTensor Function(CTensor, Pointer<FFIIndex>, int)
      >('torchffi_tensor_index');

  static final view = nativeLib
      .lookupFunction<
        CTensor Function(CTensor, Pointer<Int64>, Size),
        CTensor Function(CTensor, Pointer<Int64>, int)
      >('torchffi_tensor_view');

  static final permute = nativeLib
      .lookupFunction<
        CTensor Function(CTensor, Pointer<Int64>, Size),
        CTensor Function(CTensor, Pointer<Int64>, int)
      >('torchffi_tensor_permute');

  static final transpose = nativeLib
      .lookupFunction<
        CTensor Function(CTensor, Int64, Int64),
        CTensor Function(CTensor, int, int)
      >('torchffi_tensor_transpose');

  static final expand = nativeLib
      .lookupFunction<
        CTensor Function(CTensor, Pointer<Int64>, Size, Bool),
        CTensor Function(CTensor, Pointer<Int64>, int, bool)
      >('torchffi_tensor_expand');

  static final contiguous = nativeLib
      .lookupFunction<
        CTensor Function(CTensor, Int8),
        CTensor Function(CTensor, int)
      >('torchffi_tensor_contiguous');

  static final pad = nativeLib
      .lookupFunction<
        CTensor Function(CTensor, Pointer<Int64>, Size, Uint8, Pointer<Double>),
        CTensor Function(CTensor, Pointer<Int64>, int, int, Pointer<Double>)
      >('torchffi_tensor_pad');

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

  static final pow = nativeLib
      .lookupFunction<
        CTensor Function(CTensor, FFIScalar),
        CTensor Function(CTensor tensor, FFIScalar exponent)
      >('torchffi_tensor_pow');

  static final rsqrt = nativeLib
      .lookupFunction<
        CTensor Function(CTensor),
        CTensor Function(CTensor tensor)
      >('torchffi_tensor_rsqrt');

  static final bitwiseNot = nativeLib
      .lookupFunction<
        CTensor Function(CTensor),
        CTensor Function(CTensor tensor)
      >('torchffi_tensor_bitwise_not');

  static final bitwiseAnd = nativeLib
      .lookupFunction<
        CTensor Function(CTensor, CTensor),
        CTensor Function(CTensor tensor1, CTensor tensor2)
      >('torchffi_tensor_bitwise_and');

  static final bitwiseOr = nativeLib
      .lookupFunction<
        CTensor Function(CTensor, CTensor),
        CTensor Function(CTensor tensor1, CTensor tensor2)
      >('torchffi_tensor_bitwise_or');

  static final bitwiseXor = nativeLib
      .lookupFunction<
        CTensor Function(CTensor, CTensor),
        CTensor Function(CTensor tensor1, CTensor tensor2)
      >('torchffi_tensor_bitwise_xor');

  static final sum = nativeLib
      .lookupFunction<
        CTensor Function(CTensor, Pointer<Int64>, Size, Bool, Pointer<Uint8>),
        CTensor Function(CTensor, Pointer<Int64>, int, bool, Pointer<Uint8>)
      >('torchffi_tensor_sum');

  static final mean = nativeLib
      .lookupFunction<
        CTensor Function(CTensor, Pointer<Int64>, Size, Bool, Pointer<Uint8>),
        CTensor Function(CTensor, Pointer<Int64>, int, bool, Pointer<Uint8>)
      >('torchffi_tensor_mean');

  static final matmul = nativeLib
      .lookupFunction<
        CTensor Function(CTensor, CTensor),
        CTensor Function(CTensor tensor1, CTensor tensor2)
      >('torchffi_tensor_matmul');

  static final sigmoid = nativeLib
      .lookupFunction<
        CTensor Function(CTensor),
        CTensor Function(CTensor tensor)
      >('torchffi_tensor_sigmoid');

  static final relu = nativeLib
      .lookupFunction<
        CTensor Function(CTensor),
        CTensor Function(CTensor tensor)
      >('torchffi_tensor_relu');

  static final gelu = nativeLib
      .lookupFunction<
        CTensor Function(CTensor, Pointer<Utf8>),
        CTensor Function(CTensor tensor, Pointer<Utf8>)
      >('torchffi_tensor_gelu');

  static final silu = nativeLib
      .lookupFunction<
        CTensor Function(CTensor),
        CTensor Function(CTensor tensor)
      >('torchffi_tensor_silu');

  static final linear = nativeLib
      .lookupFunction<
        CTensor Function(CTensor, CTensor, CTensor),
        CTensor Function(CTensor input, CTensor weight, CTensor bias)
      >('torchffi_linear');

  static final layerNorm = nativeLib
      .lookupFunction<
        CTensor Function(
          CTensor,
          Pointer<Int64>,
          Size,
          CTensor weight,
          CTensor bias,
          Double eps,
          Bool,
        ),
        CTensor Function(
          CTensor,
          Pointer<Int64>,
          int,
          CTensor,
          CTensor,
          double,
          bool,
        )
      >('torchffi_layer_norm');

  static final groupNorm = nativeLib
      .lookupFunction<
        CTensor Function(CTensor, Int64, CTensor, CTensor, Double),
        CTensor Function(CTensor, int, CTensor, CTensor, double)
      >('torchffi_group_norm');

  static final dropout = nativeLib
      .lookupFunction<
        CTensor Function(CTensor, Double, Bool),
        CTensor Function(CTensor, double, bool)
      >('torchffi_dropout');

  static final softmax = nativeLib
      .lookupFunction<
        CTensor Function(CTensor, Int64, Pointer<Int8>),
        CTensor Function(CTensor, int, Pointer<Int8>)
      >('torchffi_softmax');

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

  static final conv2d = nativeLib
      .lookupFunction<
        CTensor Function(
          CTensor,
          CTensor,
          CTensor,
          Pointer<Int64>,
          Pointer<Int64>,
          Pointer<Int64>,
          Int64,
        ),
        CTensor Function(
          CTensor input,
          CTensor weight,
          CTensor bias,
          Pointer<Int64> stride,
          Pointer<Int64> padding,
          Pointer<Int64> dilation,
          int groups,
        )
      >('torchffi_conv2d');

  static final upsampleNearest = nativeLib
      .lookupFunction<
        CTensor Function(CTensor, Pointer<Int64>, Size),
        CTensor Function(
          CTensor input,
          Pointer<Int64> outputSize,
          int outputSizeLen,
        )
      >('torchffi_upsample_nearest');

  static final upsampleNearestScale = nativeLib
      .lookupFunction<
        CTensor Function(CTensor, Pointer<Double>, Size),
        CTensor Function(
          CTensor input,
          Pointer<Double> outputSize,
          int outputSizeLen,
        )
      >('torchffi_upsample_nearest_scale');

  static final upsampleNearestExact = nativeLib
      .lookupFunction<
        CTensor Function(CTensor, Pointer<Int64>, Size),
        CTensor Function(
          CTensor input,
          Pointer<Int64> outputSize,
          int outputSizeLen,
        )
      >('torchffi_upsample_nearest_exact');

  static final upsampleNearestExactScale = nativeLib
      .lookupFunction<
        CTensor Function(CTensor, Pointer<Double>, Size),
        CTensor Function(
          CTensor input,
          Pointer<Double> outputSize,
          int outputSizeLen,
        )
      >('torchffi_upsample_nearest_exact_scale');

  static final avgPool2D = nativeLib
      .lookupFunction<
        CTensor Function(
          CTensor,
          Int64 kernelSizeH,
          Int64 kernelSizeW,
          Int64 strideH,
          Int64 strideW,
          Int64 paddingH,
          Int64 paddingW,
          Bool ceilMode,
          Bool countIncludePad,
          Pointer<Int64> divisorOverride,
        ),
        CTensor Function(
          CTensor input,
          int kernelSizeH,
          int kernelSizeW,
          int strideH,
          int strideW,
          int paddingH,
          int paddingW,
          bool ceilMode,
          bool countIncludePad,
          Pointer<Int64> divisorOverride,
        )
      >('torchffi_avg_pool2d');
}
