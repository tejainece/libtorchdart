import 'dart:ffi';

import 'package:ffi/ffi.dart';
import 'package:libtorchdart/libtorchdart.dart';
import 'package:libtorchdart/src/torch_ffi/torch_ffi.dart';

typedef CTensor = Pointer<Void>;

enum FFIIndexType {
  newDimType,
  ellipsisType,
  intType,
  boolType,
  sliceType,
  tensorType,
}

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

abstract class FFITensor {
  static final constructor = nativeLib
      .lookupFunction<CTensor Function(), CTensor Function()>(
        'torchffi_tensor_new',
      );

  static final delete = nativeLib
      .lookup<NativeFunction<Void Function(Pointer<Void>)>>(
        'torchffi_tensor_delete',
      );

  static void deleteTensor(CTensor tensor) {
    delete.asFunction<void Function(CTensor)>()(tensor);
  }

  static final empty = nativeLib
      .lookupFunction<
        CTensor Function(Pointer<Int64>, Size, CTensorOptions),
        CTensor Function(Pointer<Int64>, int dims, CTensorOptions)
      >('torchffi_tensor_new_empty');

  static final zeros = nativeLib
      .lookupFunction<
        CTensor Function(Pointer<Int64>, Size, CTensorOptions),
        CTensor Function(Pointer<Int64>, int dims, CTensorOptions)
      >('torchffi_tensor_new_zeros');

  static final ones = nativeLib
      .lookupFunction<
        CTensor Function(Pointer<Int64>, Size, CTensorOptions),
        CTensor Function(Pointer<Int64>, int dims, CTensorOptions)
      >('torchffi_tensor_new_ones');

  static final arange = nativeLib
      .lookupFunction<
        CTensor Function(Int64 end, CTensorOptions),
        CTensor Function(int end, CTensorOptions)
      >('torchffi_tensor_new_arange');

  static final rand = nativeLib
      .lookupFunction<
        CTensor Function(Pointer<Int64>, Size, CGenerator, CTensorOptions),
        CTensor Function(Pointer<Int64>, int dims, CGenerator, CTensorOptions)
      >('torchffi_tensor_new_rand');

  static final randn = nativeLib
      .lookupFunction<
        CTensor Function(Pointer<Int64>, Size, CGenerator, CTensorOptions),
        CTensor Function(Pointer<Int64>, int dims, CGenerator, CTensorOptions)
      >('torchffi_tensor_new_randn');

  static final eye = nativeLib
      .lookupFunction<
        CTensor Function(Int64, Int64, CTensorOptions),
        CTensor Function(int n, int m, CTensorOptions options)
      >('torchffi_tensor_new_eye');

  static final fromBlob = nativeLib
      .lookupFunction<
        CTensor Function(
          Pointer<Void>,
          Pointer<Int64>,
          Size dims,
          CTensorOptions,
        ),
        CTensor Function(
          Pointer<Void>,
          Pointer<Int64>,
          int dims,
          CTensorOptions,
        )
      >('torchffi_tensor_new_from_blob');

  static final dataPointer = nativeLib
      .lookupFunction<
        Pointer<Void> Function(CTensor),
        Pointer<Void> Function(CTensor tensor)
      >('torchffi_tensor_data_pointer');

  static final ones_ = nativeLib
      .lookupFunction<Void Function(CTensor), void Function(CTensor)>(
        'torchffi_tensor_ones_',
      );

  static final zeros_ = nativeLib
      .lookupFunction<Void Function(CTensor), void Function(CTensor)>(
        'torchffi_tensor_zeros_',
      );

  static final eye_ = nativeLib
      .lookupFunction<Void Function(CTensor), void Function(CTensor)>(
        'torchffi_tensor_eye_',
      );

  static final fill_ = nativeLib
      .lookupFunction<
        Void Function(CTensor, CScalar),
        void Function(CTensor, CScalar)
      >('torchffi_tensor_fill_');

  static final rand_ = nativeLib
      .lookupFunction<
        Void Function(CTensor, CGenerator),
        void Function(CTensor, CGenerator)
      >('torchffi_tensor_rand_');

  static final normal_ = nativeLib
      .lookupFunction<
        Void Function(CTensor, CGenerator, Double, Double),
        void Function(CTensor, CGenerator, double, double)
      >('torchffi_tensor_normal_');

  static final uniform_ = nativeLib
      .lookupFunction<
        CTensor Function(CTensor, CGenerator, Double, Double),
        CTensor Function(CTensor, CGenerator, double, double)
      >('torchffi_tensor_uniform_');

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
        CDevice Function(CTensor),
        CDevice Function(CTensor tensor)
      >('torchffi_tensor_device');

  static final scalar = nativeLib
      .lookupFunction<
        CScalar Function(CTensor),
        CScalar Function(CTensor tensor)
      >('torchffi_tensor_scalar');

  static final scalarAt = nativeLib
      .lookupFunction<
        CScalar Function(CTensor, Int64),
        CScalar Function(CTensor, int)
      >('torchffi_tensor_scalar_at');

  static final get = nativeLib
      .lookupFunction<
        CTensor Function(CTensor, Int64),
        CTensor Function(CTensor tensor, int index)
      >('torchffi_tensor_get');

  static final datatype = nativeLib
      .lookupFunction<Int8 Function(CTensor), int Function(CTensor tensor)>(
        'torchffi_tensor_get_datatype',
      );

  static final to = nativeLib
      .lookupFunction<
        CTensor Function(CTensor, CTensorOptions, Bool, Bool),
        CTensor Function(
          CTensor tensor,
          CTensorOptions,
          bool nonBlocking,
          bool copy,
        )
      >('torchffi_tensor_to');

  static final copy_ = nativeLib
      .lookupFunction<
        Void Function(CTensor, CTensor, Bool),
        void Function(CTensor, CTensor, bool)
      >('torchffi_tensor_copy_');

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

  static final reshape = nativeLib
      .lookupFunction<
        CTensor Function(CTensor, Pointer<Int64>, Size),
        CTensor Function(CTensor, Pointer<Int64>, int)
      >('torchffi_tensor_reshape');

  static final flatten = nativeLib
      .lookupFunction<
        CTensor Function(CTensor, Int64 start, Int64 end),
        CTensor Function(CTensor, int start, int end)
      >('torchffi_tensor_flatten');

  static final splitEqually = nativeLib
      .lookupFunction<
        Pointer<CTensor> Function(CTensor, Int64, Int64),
        Pointer<CTensor> Function(CTensor, int, int)
      >('torchffi_tensor_split_equally');

  static final split = nativeLib
      .lookupFunction<
        Pointer<CTensor> Function(CTensor, Pointer<Int64>, Size, Int64),
        Pointer<CTensor> Function(CTensor, Pointer<Int64>, int, int)
      >('torchffi_tensor_split');

  static final chunk = nativeLib
      .lookupFunction<
        Pointer<CTensor> Function(CTensor, Int64, Int64),
        Pointer<CTensor> Function(CTensor, int chunks, int dim)
      >('torchffi_tensor_chunk');

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

  static final squeeze = nativeLib
      .lookupFunction<
        CTensor Function(CTensor, Pointer<Int64>),
        CTensor Function(CTensor, Pointer<Int64>)
      >('torchffi_tensor_squeeze');

  static final unsqueeze = nativeLib
      .lookupFunction<
        CTensor Function(CTensor, Int64),
        CTensor Function(CTensor, int)
      >('torchffi_tensor_unsqueeze');

  static final elementSize = nativeLib
      .lookupFunction<Int64 Function(CTensor), int Function(CTensor)>(
        'torchffi_tensor_element_size',
      );

  static final pad = nativeLib
      .lookupFunction<
        CTensor Function(CTensor, Pointer<Int64>, Size, Uint8, Pointer<Double>),
        CTensor Function(CTensor, Pointer<Int64>, int, int, Pointer<Double>)
      >('torchffi_tensor_pad');

  static final allClose = nativeLib
      .lookupFunction<
        Bool Function(
          CTensor,
          CTensor,
          Double rtol,
          Double atol,
          Bool equalNan,
        ),
        bool Function(
          CTensor tensor1,
          CTensor tensor2,
          double rtol,
          double atol,
          bool equalNan,
        )
      >('torchffi_tensor_allclose');

  static final addition = nativeLib
      .lookupFunction<
        CTensor Function(CTensor, CTensor, CScalar alpha),
        CTensor Function(CTensor tensor1, CTensor tensor2, CScalar alpha)
      >('torchffi_tensor_addition');

  static final subtraction = nativeLib
      .lookupFunction<
        CTensor Function(CTensor, CTensor, CScalar alpha),
        CTensor Function(CTensor tensor1, CTensor tensor2, CScalar alpha)
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

  static final divisionScalar = nativeLib
      .lookupFunction<
        CTensor Function(CTensor, CScalar),
        CTensor Function(CTensor tensor, CScalar scalar)
      >('torchffi_tensor_division_scalar');

  static final clone = nativeLib
      .lookupFunction<
        CTensor Function(CTensor, Pointer<Int8>),
        CTensor Function(CTensor tensor, Pointer<Int8> memoryFormat)
      >('torchffi_tensor_clone');

  static final pow = nativeLib
      .lookupFunction<
        CTensor Function(CTensor, CScalar),
        CTensor Function(CTensor tensor, CScalar exponent)
      >('torchffi_tensor_pow');

  static final rsqrt = nativeLib
      .lookupFunction<
        CTensor Function(CTensor),
        CTensor Function(CTensor tensor)
      >('torchffi_tensor_rsqrt');

  static final sin = nativeLib
      .lookupFunction<
        CTensor Function(CTensor),
        CTensor Function(CTensor tensor)
      >('torchffi_tensor_sin');

  static final cos = nativeLib
      .lookupFunction<
        CTensor Function(CTensor),
        CTensor Function(CTensor tensor)
      >('torchffi_tensor_cos');

  static final exp = nativeLib
      .lookupFunction<
        CTensor Function(CTensor),
        CTensor Function(CTensor tensor)
      >('torchffi_tensor_exp');

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

  static final softmax = nativeLib
      .lookupFunction<
        CTensor Function(CTensor, Int64, Pointer<Int8>),
        CTensor Function(CTensor, int, Pointer<Int8>)
      >('torchffi_softmax');

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

  static final cat = nativeLib
      .lookupFunction<
        CTensor Function(Pointer<CTensor>, Int64, Int64),
        CTensor Function(Pointer<CTensor> tensors, int length, int dim)
      >('torchffi_cat');
}

abstract class FFINN {
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

  static final rmsNorm = nativeLib
      .lookupFunction<
        CTensor Function(
          CTensor,
          Pointer<Int64>,
          Size,
          CTensor,
          Pointer<Double>,
        ),
        CTensor Function(
          CTensor,
          Pointer<Int64>,
          int,
          CTensor,
          Pointer<Double> eps,
        )
      >('torchffi_rms_norm');

  static final dropout = nativeLib
      .lookupFunction<
        CTensor Function(CTensor, Double, Bool),
        CTensor Function(CTensor, double, bool)
      >('torchffi_dropout');

  static final dropout_ = nativeLib
      .lookupFunction<
        Void Function(CTensor, Double, Bool),
        void Function(CTensor, double, bool)
      >('torchffi_dropout_');

  static final embeddingRenorm_ = nativeLib
      .lookupFunction<
        CTensor Function(CTensor, CTensor, Double, Double),
        CTensor Function(
          CTensor weights,
          CTensor indices,
          double maxNorm,
          double normType,
        )
      >('torchffi_embedding_renorm_');

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

abstract class FFINN2D {
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
