import 'dart:math';

import 'package:libtorchdart/libtorchdart.dart';
import 'package:libtorchdart/src/nn/pooling.dart';

/// A 2D upsampling layer with an optional convolution.
class Upsample2D extends Module {
  final Normalization? norm;
  final Conv2D? conv;
  final bool useConvTransposed;
  final bool interpolate;

  Upsample2D({
    required this.norm,
    required this.conv,
    this.useConvTransposed = false,
    this.interpolate = true,
  });

  Tensor forward(Tensor hiddenStates, {SymmetricPadding2D? outputSize}) {
    if (norm != null) {
      hiddenStates = norm!.forward(hiddenStates.permute([0, 2, 3, 1])).permute([
        0,
        3,
        1,
        2,
      ]);
    }

    if (useConvTransposed) {
      return conv!.forward(hiddenStates);
    }

    if (hiddenStates.shape[0] >= 64) {
      hiddenStates = hiddenStates.contiguous();
    }

    if (interpolate) {
      if (outputSize != null) {
        final scale = [
          outputSize.vertical / hiddenStates.shape[2],
          outputSize.horizontal / hiddenStates.shape[3],
        ].reduce(max);

        if (hiddenStates.numel * scale > 2 >> 31) {
          hiddenStates = hiddenStates.contiguous();
        }

        hiddenStates = interpolateNearest(hiddenStates, outputSize.to2List());
      } else {
        if (hiddenStates.numel * 2 > 2 >> 31) {
          hiddenStates = hiddenStates.contiguous();
        }

        hiddenStates = interpolateNearestScale(hiddenStates, [2.0, 2.0]);
      }
    }

    if (conv != null) {
      hiddenStates = conv!.forward(hiddenStates);
    }
    return hiddenStates;
  }

  static Future<Upsample2D> loadFromSafeTensor(
    SafeTensorLoader loader, {
    String prefix = '',
    bool useConvTransposed = false,
    bool interpolate = true,
    SymmetricPadding2D padding = const SymmetricPadding2D(
      vertical: 1,
      horizontal: 1,
    ),
  }) async {
    Normalization? norm;
    Conv2D? conv;
    if (useConvTransposed) {
      throw UnimplementedError();
    } else {
      if (loader.hasTensor('${prefix}conv')) {
        conv = await Conv2D.loadFromSafeTensor(
          loader,
          prefix: '${prefix}conv',
          padding: padding,
        );
      }
    }

    return Upsample2D(
      conv: conv,
      norm: norm,
      useConvTransposed: useConvTransposed,
      interpolate: interpolate,
    );
  }

  static Upsample2D make({
    required int numChannels,
    int? numOutChannels,
    bool useConv = false,
    bool useConvTransposed = false,
    SymmetricPadding2D? kernelSize,
    SymmetricPadding2D padding = const SymmetricPadding2D.same(1),
    SamplerNormalizationConfig? normConfig,
    bool hasBias = true,
    bool interpolate = true,
  }) {
    Normalization? norm; // TODO

    Conv2D conv;
    if (useConvTransposed) {
      kernelSize ??= SymmetricPadding2D.same(4);

      throw UnimplementedError();
    } else {
      kernelSize ??= SymmetricPadding2D.same(3);
      conv = Conv2D.make(
        numInChannels: numChannels,
        numOutChannels: numOutChannels ?? numChannels,
        kernelSize: kernelSize,
        padding: padding,
        hasBias: hasBias,
      );
    }

    return Upsample2D(
      conv: conv,
      norm: norm,
      useConvTransposed: useConvTransposed,
      interpolate: interpolate,
    );
  }
}

class SamplerNormalizationConfig {
  // TODO norm type

  final double? eps;

  final bool isElementwiseAffine;

  SamplerNormalizationConfig({this.eps, this.isElementwiseAffine = false});
}

class Downsample2D {
  final Normalization? norm;
  final SimpleModule? conv;

  Downsample2D({this.norm, this.conv});

  Tensor forward(Tensor hiddenStates) {
    if (norm != null) {
      hiddenStates = norm!.forward(hiddenStates.permute([0, 2, 3, 1])).permute([
        0,
        3,
        1,
        2,
      ]);
    }

    if (conv is Conv2D &&
        (conv as Conv2D).padding == SymmetricPadding2D.same(0)) {
      hiddenStates = hiddenStates.pad([0, 1, 0, 1], value: 0);
    }

    if (conv != null) {
      hiddenStates = conv!.forward(hiddenStates);
    }
    return hiddenStates;
  }

  static Future<Downsample2D> loadFromSafeTensor(
    SafeTensorLoader loader, {
    String prefix = '',
    SymmetricPadding2D padding = const SymmetricPadding2D(
      vertical: 1,
      horizontal: 1,
    ),
  }) async {
    final stride = SymmetricPadding2D.same(2);
    Normalization? norm;
    SimpleModule? conv;
    if (loader.hasTensor('${prefix}conv')) {
      conv = await Conv2D.loadFromSafeTensor(
        loader,
        prefix: '${prefix}conv',
        padding: padding,
        stride: stride,
      );
    } else {
      conv = AvgPool2D(kernelSize: stride, stride: stride);
    }
    return Downsample2D(norm: norm, conv: conv);
  }

  static Downsample2D make({
    required int numChannels,
    int? numOutChannels,
    bool useConv = false,
    SymmetricPadding2D kernelSize = const SymmetricPadding2D.same(3),
    SymmetricPadding2D padding = const SymmetricPadding2D.same(1),
    bool hasBias = true,
  }) {
    final stride = SymmetricPadding2D.same(2);
    Normalization? norm;
    SimpleModule conv;
    if (useConv) {
      conv = Conv2D.make(
        numInChannels: numChannels,
        numOutChannels: numOutChannels ?? numChannels,
        kernelSize: kernelSize,
        stride: stride,
        padding: padding,
        hasBias: true,
      );
    } else {
      conv = AvgPool2D(kernelSize: stride, stride: stride);
    }
    return Downsample2D(norm: norm, conv: conv);
  }
}
