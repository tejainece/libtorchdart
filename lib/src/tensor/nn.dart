import 'package:tensor/tensor.dart';
import 'package:tensor/src/ffi/torch_ffi.dart';

import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart' as ffi;

abstract class NNUtil {
  static Tensor layerNorm(
    Tensor input,
    List<int> normalizedShape, {
    Tensor? weight,
    Tensor? bias,
    double eps = 1e-5,
  }) {
    final arena = ffi.Arena();
    try {
      final normalizedShapePointer = arena.allocate<ffi.Int64>(
        ffi.sizeOf<ffi.Int64>() * normalizedShape.length,
      );
      normalizedShapePointer
          .asTypedList(normalizedShape.length)
          .setAll(0, normalizedShape);

      final tensorPtr = FFINN.layerNorm(
        input.nativePtr,
        normalizedShapePointer,
        normalizedShape.length,
        weight?.nativePtr ?? ffi.nullptr,
        bias?.nativePtr ?? ffi.nullptr,
        eps,
        true, // TODO: enable_cudnn
      );
      return Tensor(tensorPtr);
    } finally {
      arena.releaseAll();
    }
  }

  static Tensor groupNorm(
    Tensor input,
    int numGroups, {
    Tensor? weight,
    Tensor? bias,
    double eps = 1e-5,
  }) {
    final tensorPtr = FFINN.groupNorm(
      input.nativePtr,
      numGroups,
      weight?.nativePtr ?? ffi.nullptr,
      bias?.nativePtr ?? ffi.nullptr,
      eps,
    );
    return Tensor(tensorPtr);
  }

  static Tensor rmsNorm(
    Tensor input,
    List<int> normalizedShape, {
    Tensor? weight,
    double? eps,
  }) {
    final arena = ffi.Arena();
    try {
      final normalizedShapePointer = arena.allocate<ffi.Int64>(
        ffi.sizeOf<ffi.Int64>() * normalizedShape.length,
      );
      normalizedShapePointer
          .asTypedList(normalizedShape.length)
          .setAll(0, normalizedShape);

      ffi.Pointer<ffi.Double> epsPointer = ffi.nullptr;
      if (eps != null) {
        epsPointer = arena.allocate<ffi.Double>(ffi.sizeOf<ffi.Double>())
          ..value = eps;
      }

      final tensorPtr = FFINN.rmsNorm(
        input.nativePtr,
        normalizedShapePointer,
        normalizedShape.length,
        weight?.nativePtr ?? ffi.nullptr,
        epsPointer,
      );
      return Tensor(tensorPtr);
    } finally {
      arena.releaseAll();
    }
  }

  static Tensor dropout(Tensor input, double p, {bool training = true}) {
    final output = FFINN.dropout(input.nativePtr, p, training);
    return Tensor(output);
  }

  static void dropout_(Tensor input, double p, {bool training = true}) {
    FFINN.dropout_(input.nativePtr, p, training);
  }

  static Tensor linear(Tensor input, Tensor weight, {Tensor? bias}) {
    final tensorPtr = FFINN.linear(
      input.nativePtr,
      weight.nativePtr,
      bias?.nativePtr ?? ffi.nullptr,
    );
    return Tensor(tensorPtr);
  }

  static Tensor embeddingRenorm_(
    Tensor weights,
    Tensor indices,
    double maxNorm,
    double normType,
  ) {
    final tensorPtr = FFINN.embeddingRenorm_(
      weights.nativePtr,
      indices.nativePtr,
      maxNorm,
      normType,
    );
    return Tensor(tensorPtr);
  }

  static Tensor embedding(
    Tensor weights,
    Tensor indices, {
    int? paddingIdx,
    bool scaleGradByFreq = false,
    bool sparse = false,
    ({double maxNorm, double normType})? norm,
  }) {
    if (paddingIdx == null) {
      paddingIdx = -1;
    } else {
      if (paddingIdx < 0) {
        paddingIdx = weights.shape[0] + paddingIdx;
      }
    }

    if (norm != null) {
      weights = weights.contiguous();
      embeddingRenorm_(weights, indices, norm.maxNorm, norm.normType);
    }

    final tensorPtr = FFINN.embedding(
      weights.nativePtr,
      indices.nativePtr,
      paddingIdx,
      scaleGradByFreq,
      sparse,
    );

    return Tensor(tensorPtr);
  }
}

abstract class NN2DUtil {
  static Tensor conv2d(
    Tensor input,
    Tensor weight, {
    Tensor? bias,
    SymmetricPadding2D stride = const SymmetricPadding2D.same(1),
    SymmetricPadding2D padding = const SymmetricPadding2D.same(0),
    SymmetricPadding2D dilation = const SymmetricPadding2D.same(1),
    int groups = 1,
  }) {
    final arena = ffi.Arena();
    try {
      ffi.Pointer<ffi.Int64> stridePointer = arena.allocate<ffi.Int64>(
        ffi.sizeOf<ffi.Int64>() * 2,
      );
      stridePointer.value = stride.vertical;
      (stridePointer + 1).value = stride.horizontal;

      ffi.Pointer<ffi.Int64> paddingPointer = arena.allocate<ffi.Int64>(
        ffi.sizeOf<ffi.Int64>() * 2,
      );
      paddingPointer.value = padding.vertical;
      (paddingPointer + 1).value = padding.horizontal;

      ffi.Pointer<ffi.Int64> dilationPointer = arena.allocate<ffi.Int64>(
        ffi.sizeOf<ffi.Int64>() * 2,
      );
      dilationPointer.value = dilation.vertical;
      (dilationPointer + 1).value = dilation.horizontal;

      final tensorPtr = FFINN2D.conv2d(
        input.nativePtr,
        weight.nativePtr,
        bias?.nativePtr ?? ffi.nullptr,
        stridePointer,
        paddingPointer,
        dilationPointer,
        groups,
      );
      return Tensor(tensorPtr);
    } finally {
      arena.releaseAll();
    }
  }

  static Tensor conv2dTranspose(
    Tensor input,
    Tensor weight, {
    Tensor? bias,
    SymmetricPadding2D stride = const SymmetricPadding2D.same(1),
    SymmetricPadding2D padding = const SymmetricPadding2D.same(0),
    SymmetricPadding2D outputPadding = const SymmetricPadding2D.same(0),
    SymmetricPadding2D dilation = const SymmetricPadding2D.same(1),
    int groups = 1,
  }) {
    final arena = ffi.Arena();
    try {
      ffi.Pointer<ffi.Int64> stridePointer = arena.allocate<ffi.Int64>(
        ffi.sizeOf<ffi.Int64>() * 2,
      );
      stridePointer.value = stride.vertical;
      (stridePointer + 1).value = stride.horizontal;

      ffi.Pointer<ffi.Int64> paddingPointer = arena.allocate<ffi.Int64>(
        ffi.sizeOf<ffi.Int64>() * 2,
      );
      paddingPointer.value = padding.vertical;
      (paddingPointer + 1).value = padding.horizontal;

      ffi.Pointer<ffi.Int64> outputPaddingPointer = arena.allocate<ffi.Int64>(
        ffi.sizeOf<ffi.Int64>() * 2,
      );
      outputPaddingPointer.value = outputPadding.vertical;
      (outputPaddingPointer + 1).value = outputPadding.horizontal;

      ffi.Pointer<ffi.Int64> dilationPointer = arena.allocate<ffi.Int64>(
        ffi.sizeOf<ffi.Int64>() * 2,
      );
      dilationPointer.value = dilation.vertical;
      (dilationPointer + 1).value = dilation.horizontal;

      final tensorPtr = FFINN2D.conv2dTranspose(
        input.nativePtr,
        weight.nativePtr,
        bias?.nativePtr ?? ffi.nullptr,
        stridePointer,
        paddingPointer,
        outputPaddingPointer,
        dilationPointer,
        groups,
      );
      return Tensor(tensorPtr);
    } finally {
      arena.releaseAll();
    }
  }

  static Tensor avgPool2D(
    Tensor input,
    SymmetricPadding2D kernelSize, {

    /// If null, it is set to [kernelSize]
    SymmetricPadding2D? stride,
    SymmetricPadding2D padding = const SymmetricPadding2D(
      vertical: 0,
      horizontal: 0,
    ),
    bool ceilMode = false,
    bool countIncludePad = true,
    int? divisorOverride,
  }) {
    stride ??= kernelSize;
    final arena = ffi.Arena();
    try {
      ffi.Pointer<ffi.Int64> divisorOverridePointer = ffi.nullptr;
      if (divisorOverride != null) {
        divisorOverridePointer = arena.allocate<ffi.Int64>(
          ffi.sizeOf<ffi.Int64>(),
        );
        divisorOverridePointer.value = divisorOverride;
      }

      final tensorPtr = FFINN2D.avgPool2D(
        input.nativePtr,
        kernelSize.vertical,
        kernelSize.horizontal,
        stride.vertical,
        stride.horizontal,
        padding.vertical,
        padding.horizontal,
        ceilMode,
        countIncludePad,
        divisorOverridePointer,
      );
      return Tensor(tensorPtr);
    } finally {
      arena.releaseAll();
    }
  }
}
