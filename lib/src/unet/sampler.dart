import 'dart:math';

import 'package:libtorchdart/libtorchdart.dart';
import 'package:libtorchdart/src/nn/pooling.dart';

/// A 2D upsampling layer with an optional convolution.
class Upsample2D extends Module implements SimpleModule {
  final Normalization? norm;
  final Conv2DInterface? conv;
  final bool interpolate;

  Upsample2D({
    super.name = 'upsample',
    required this.norm,
    required this.conv,
    this.interpolate = true,
  }) {
    if (norm != null && conv != null) {
      if (norm is LayerNorm) {
        assert((norm as LayerNorm).normalizedShape.length == 1);
        assert((norm as LayerNorm).normalizedShape[0] == conv!.numInChannels);
      }
    }
  }

  @override
  Tensor forward(
    Tensor hiddenStates, {
    SymmetricPadding2D? outputSize,
    required Context context,
  }) {
    if (numInChannels != null) {
      assert(hiddenStates.shape[1] == numInChannels);
    }

    if (norm != null) {
      hiddenStates = norm!
          .forward(hiddenStates.permute([0, 2, 3, 1]), context: context)
          .permute([0, 3, 1, 2]);
    }

    if (useConvTransposed) {
      return conv!.forward(hiddenStates, context: context);
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
      hiddenStates = conv!.forward(hiddenStates, context: context);
    }
    return hiddenStates;
  }

  @override
  void resetParameters() {
    conv?.resetParameters();
    norm?.resetParameters();
  }

  @override
  late final Iterable<Tensor> parameters = [];

  @override
  late final Iterable<Module> submodules = [
    if (norm != null) norm!,
    if (conv != null) conv!,
  ];

  bool get useConvTransposed => conv is Conv2DTranspose;

  int? get numInChannels {
    if (norm != null) {
      if (norm is LayerNorm) {
        return (norm as LayerNorm).normalizedShape[0];
      }
    }
    if (conv != null) {
      return conv!.numInChannels;
    }
    return null;
  }

  @override
  Map<String, dynamic> get meta => {
    "useConvTransposed": useConvTransposed,
    "interpolate": interpolate,
    "norm": norm?.meta,
    "conv": conv?.meta,
  };

  static Future<Upsample2D> loadFromSafeTensor(
    SafeTensorLoader loader, {
    String prefix = '',
    String name = 'upsample',
    required int numChannels,
    bool useConvTransposed = false,
    bool interpolate = true,
    SymmetricPadding2D padding = const SymmetricPadding2D.same(1),
    SamplerNormalizationConfig? normConfig,
    String convName = 'conv',
    String normName = 'norm',
  }) async {
    Normalization? norm;
    if (normConfig != null) {
      if (normConfig.normType == .lnNorm) {
        norm = await LayerNorm.loadFromSafeTensor(
          loader,
          prefix: '$prefix$normName.',
          name: normName,
          normalizedShape: [numChannels],
          eps: normConfig.eps,
        );
      } else if (normConfig.normType == .rmsNorm) {
        norm = await RMSNormWithBias.loadFromSafeTensor(
          loader,
          prefix: '$prefix$normName.',
          eps: normConfig.eps,
        );
      } else {
        throw UnimplementedError(
          'Unknown Upsampler2D normalization type: ${normConfig.normType}',
        );
      }
    }

    Conv2DInterface? conv;
    if (useConvTransposed) {
      if (loader.hasTensorWithPrefix('$prefix$convName')) {
        conv = await Conv2DTranspose.loadFromSafeTensor(
          loader,
          prefix: '$prefix$convName.',
          name: convName,
          padding: padding,
        );
        assert(numChannels == (conv as Conv2DTranspose).numInChannels);
      }
    } else {
      if (loader.hasTensorWithPrefix('$prefix$convName')) {
        conv = await Conv2D.loadFromSafeTensor(
          loader,
          prefix: '$prefix$convName.',
          name: convName,
          padding: padding,
        );
        assert(numChannels == conv.numInChannels);
      }
    }

    return Upsample2D(
      name: name,
      conv: conv,
      norm: norm,
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
    String name = 'upsample',
    String convName = 'conv',
    String normName = 'norm',
  }) {
    Normalization? norm;
    if (normConfig != null) {
      if (normConfig.normType == .lnNorm) {
        norm = LayerNorm.make(
          name: normName,
          normalizedShape: [numChannels],
          isElementwiseAffine: normConfig.isElementwiseAffine,
        );
      } else if (normConfig.normType == .rmsNorm) {
        norm = RMSNormWithBias.make(
          name: normName,
          normalizedShape: [numChannels],
          isElementwiseAffine: normConfig.isElementwiseAffine,
        );
      } else {
        throw UnimplementedError(
          'Unknown Upsampler2D normalization type: ${normConfig.normType}',
        );
      }
    }

    Conv2DInterface? conv;
    if (useConvTransposed) {
      kernelSize ??= SymmetricPadding2D.same(4);
      conv = Conv2DTranspose.make(
        name: convName,
        numInChannels: numChannels,
        numOutChannels: numOutChannels ?? numChannels,
        kernelSize: kernelSize,
        stride: SymmetricPadding2D.same(2),
        padding: padding,
        hasBias: hasBias,
      );
    } else if (useConv) {
      kernelSize ??= SymmetricPadding2D.same(3);
      conv = Conv2D.make(
        name: convName,
        numInChannels: numChannels,
        numOutChannels: numOutChannels ?? numChannels,
        kernelSize: kernelSize,
        padding: padding,
        hasBias: hasBias,
      );
    }

    return Upsample2D(
      name: name,
      conv: conv,
      norm: norm,
      interpolate: interpolate,
    );
  }
}

/// A 2D downsampling layer with an optional convolution.
class DownSample2D extends Module implements SimpleModule {
  final Normalization? norm;
  final SimpleModule? conv;

  DownSample2D({super.name = 'downsample', this.norm, this.conv});

  @override
  Tensor forward(Tensor hiddenStates, {required Context context}) {
    if (norm != null) {
      hiddenStates = norm!
          .forward(hiddenStates.permute([0, 2, 3, 1]), context: context)
          .permute([0, 3, 1, 2]);
    }

    if (conv is Conv2D &&
        (conv as Conv2D).padding == SymmetricPadding2D.same(0)) {
      hiddenStates = hiddenStates.pad([0, 1, 0, 1], value: 0);
    }

    if (conv != null) {
      hiddenStates = conv!.forward(hiddenStates, context: context);
    }
    return hiddenStates;
  }

  @override
  void resetParameters() {
    conv?.resetParameters();
    norm?.resetParameters();
  }

  @override
  Map<String, dynamic> get meta => {"norm": norm?.meta, "conv": conv?.meta};

  @override
  late final Iterable<Tensor> parameters = [];

  @override
  late final Iterable<Module> submodules = [
    if (norm != null) norm!,
    if (conv != null) conv!,
  ];

  static Future<DownSample2D> loadFromSafeTensor(
    SafeTensorLoader loader, {
    String prefix = '',
    SymmetricPadding2D padding = const SymmetricPadding2D.same(1),
    SamplerNormalizationConfig? normConfig,
    required int numChannels,
    String name = 'downsample',
    String convName = 'conv',
    String normName = 'norm',
  }) async {
    final stride = SymmetricPadding2D.same(2);
    Normalization? norm;
    if (normConfig != null) {
      if (normConfig.normType == .lnNorm) {
        norm = await LayerNorm.loadFromSafeTensor(
          loader,
          prefix: '$prefix$normName.',
          name: normName,
          normalizedShape: [numChannels],
        );
      } else if (normConfig.normType == .rmsNorm) {
        norm = await RMSNormWithBias.loadFromSafeTensor(
          loader,
          prefix: '$prefix$normName.',
        );
      } else {
        throw UnimplementedError(
          'Unknown Upsampler2D normalization type: ${normConfig.normType}',
        );
      }
    }

    SimpleModule? conv;
    if (loader.hasTensorWithPrefix('${prefix}conv')) {
      conv = await Conv2D.loadFromSafeTensor(
        loader,
        prefix: '$prefix$convName.',
        name: convName,
        padding: padding,
        stride: stride,
      );
      assert(numChannels == (conv as Conv2D).numInChannels);
    } else {
      conv = AvgPool2D(kernelSize: stride, stride: stride);
    }
    return DownSample2D(name: name, norm: norm, conv: conv);
  }

  static DownSample2D make({
    required int numChannels,
    int? numOutChannels,
    bool useConv = false,
    SymmetricPadding2D kernelSize = const SymmetricPadding2D.same(3),
    SymmetricPadding2D padding = const SymmetricPadding2D.same(1),
    bool hasBias = true,
    SamplerNormalizationConfig? normConfig,
    double eps = 1e-5,
    String name = 'downsample',
    String convName = 'conv',
    String normName = 'norm',
  }) {
    final stride = SymmetricPadding2D.same(2);
    Normalization? norm;
    if (normConfig != null) {
      if (normConfig.normType == .lnNorm) {
        norm = LayerNorm.make(
          name: normName,
          normalizedShape: [numChannels],
          eps: eps,
          isElementwiseAffine: normConfig.isElementwiseAffine,
        );
      } else if (normConfig.normType == .rmsNorm) {
        norm = RMSNormWithBias.make(
          name: normName,
          normalizedShape: [numChannels],
          eps: eps,
          isElementwiseAffine: normConfig.isElementwiseAffine,
        );
      } else {
        throw UnimplementedError(
          'Unknown Upsampler2D normalization type: ${normConfig.normType}',
        );
      }
    }

    SimpleModule conv;
    if (useConv) {
      conv = Conv2D.make(
        name: convName,
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
    return DownSample2D(name: name, norm: norm, conv: conv);
  }
}

enum SamplerNormType { lnNorm, rmsNorm }

class SamplerNormalizationConfig {
  final SamplerNormType normType;

  final double eps;

  final bool isElementwiseAffine;

  SamplerNormalizationConfig({
    required this.normType,
    this.eps = 1e-5,
    this.isElementwiseAffine = false,
  });
}
