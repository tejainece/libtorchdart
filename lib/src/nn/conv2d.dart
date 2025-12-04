import 'dart:math';

import 'package:libtorchdart/libtorchdart.dart';

abstract class Conv2DInterface extends SimpleModule {
  Tensor get weight;
  Tensor? get bias;

  SymmetricPadding2D get stride;

  SymmetricPadding2D? get padding;

  SymmetricPadding2D get dilation;

  int get groups;

  int get numInChannels;
  int get numOutChannels;
}

class Conv2D extends Module implements SimpleModule, Conv2DInterface {
  @override
  final Tensor weight;

  @override
  final Tensor? bias;

  @override
  final SymmetricPadding2D stride;

  /// Padding used with [padMode]
  final Conv2DPad? customPad;
  @override
  final SymmetricPadding2D? padding;
  @override
  final SymmetricPadding2D dilation;
  @override
  final int groups;

  Conv2D(
    this.weight, {
    super.name = 'conv',
    this.bias,
    this.groups = 1,
    this.stride = const SymmetricPadding2D.same(1),
    this.customPad,
    this.padding,
    this.dilation = const SymmetricPadding2D.same(1),
  }) : assert(groups > 0) {
    assert(numInChannels % groups == 0);
    assert(numOutChannels % groups == 0);
    if (customPad != null) {
      if (stride != const SymmetricPadding2D.same(1)) {
        throw UnimplementedError(
          'Custom padding with stride other than 1 is not implemented',
        );
      }
    }
  }

  factory Conv2D._(
    Tensor weight, {
    String name = 'conv',
    required Tensor? bias,
    required PadMode? padMode,
    required SymmetricPadding2D stride,
    required SymmetricPadding2D? padding,
    required SymmetricPadding2D dilation,
    required int groups,
  }) {
    Conv2DPad? customPad;
    if (padMode != null) {
      final kernelSize = SymmetricPadding2D(
        vertical: weight.shape[2],
        horizontal: weight.shape[3],
      );
      final total = dilation.multiplySymmetric(kernelSize.subtractInt(1));
      final initial = total.divideInt(2);
      Padding2D customPadding = Padding2D(
        left: initial.horizontal,
        right: total.horizontal - initial.horizontal,
        top: initial.vertical,
        bottom: total.vertical - initial.vertical,
      );
      customPad = Conv2DPad(padMode: padMode, padding: customPadding);
      padding = null;
    }

    return Conv2D(
      weight,
      name: name,
      bias: bias,
      groups: groups,
      stride: stride,
      customPad: customPad,
      padding: padding,
      dilation: dilation,
    );
  }

  @override
  Tensor forward(Tensor input, {required Context context}) {
    context.onloadModule(this);
    if (customPad == null) {
      return NN2DUtil.conv2d(
        input,
        weight,
        bias: bias,
        stride: stride,
        padding: padding!,
        dilation: dilation,
        groups: groups,
      );
    }

    input = input.pad(customPad!.padding.to4List(), mode: customPad!.padMode);
    return NN2DUtil.conv2d(
      input,
      weight,
      bias: bias,
      stride: stride,
      dilation: dilation,
      groups: groups,
    );
  }

  @override
  void resetParameters({Generator? generator}) {
    Init.kaimingUniform_(weight, a: sqrt(5), generator: generator);
    if (bias != null) {
      final fan = Init.calculateKaimingFan(weight);
      if (fan.fanIn != 0) {
        double bound = 1.0 / sqrt(fan.fanIn);
        bias!.uniform_(from: -bound, to: bound, generator: generator);
      }
    }
  }

  @override
  late final Iterable<Tensor> parameters = {weight, if (bias != null) bias!};

  @override
  final Iterable<Module> submodules = const [];

  @override
  int get numInChannels => weight.shape[1] * groups;

  @override
  int get numOutChannels => weight.shape[0] * groups;

  SymmetricPadding2D get kernelSize {
    final size = weight.shape;
    return SymmetricPadding2D(vertical: size[2], horizontal: size[3]);
  }

  @override
  late final Map<String, dynamic> meta = {
    "inChannel": numInChannels,
    "outChannel": numOutChannels,
    "kernelSize": kernelSize.to2List(),
    "stride": stride.to2List(),
    "padding": padding?.to2List(),
    "dilation": dilation.to2List(),
    "groups": groups,
  };

  @override
  String toString() =>
      'Conv2D(${meta.entries.map((e) => '${e.key}: ${e.value}').join(', ')})';

  static Future<Conv2D> loadFromSafeTensor(
    SafeTensorLoader loader, {
    String prefix = '',
    String name = 'conv',
    int groups = 1,
    SymmetricPadding2D stride = const SymmetricPadding2D.same(1),
    SymmetricPadding2D? padding,
    SymmetricPadding2D dilation = const SymmetricPadding2D.same(1),
    PadMode? padMode,
  }) async {
    final weight = await loader.loadByName('${prefix}weight');
    Tensor? bias;
    if (loader.hasTensor('${prefix}bias')) {
      bias = await loader.loadByName('${prefix}bias');
    }

    return Conv2D._(
      name: name,
      weight,
      bias: bias,
      groups: groups,
      padMode: padMode,
      padding: padding,
      stride: stride,
      dilation: dilation,
    );
  }

  static Conv2D make({
    String name = 'conv',
    required int numInChannels,
    required int numOutChannels,
    SymmetricPadding2D kernelSize = const SymmetricPadding2D.same(3),
    int groups = 1,
    SymmetricPadding2D stride = const SymmetricPadding2D.same(1),
    SymmetricPadding2D padding = const SymmetricPadding2D.same(0),
    SymmetricPadding2D dilation = const SymmetricPadding2D.same(1),
    bool hasBias = true,
    PadMode? padMode,

    /// If true, uses padding so that the output size remains same as the input size
    bool padToSame = false,
    Generator? generator,
    DataType? dataType,
    Device? device,
  }) {
    Tensor weights = Tensor.empty([
      numOutChannels,
      numInChannels,
      kernelSize.vertical,
      kernelSize.horizontal,
    ]);
    Tensor? bias;
    if (hasBias) {
      bias = Tensor.empty([numOutChannels]);
    }
    return Conv2D._(
      weights,
      name: name,
      bias: bias,
      groups: groups,
      stride: stride,
      padding: padding,
      dilation: dilation,
      padMode: padMode,
    )..resetParameters(generator: generator);
  }
}

/// Applies a 2D transposed convolution operator over an input image
/// composed of several input planes.
///
/// This module can be seen as the gradient of Conv2d with respect to its input.
/// It is also known as a fractionally-strided convolution or
/// a deconvolution (although it is not an actual deconvolution operation as it does
/// not compute a true inverse of convolution). For more information, see the visualizations
/// [here](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md) and the
/// [Deconvolutional Networks](https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf) paper.
class Conv2DTranspose extends Module implements Conv2DInterface {
  @override
  final Tensor weight;
  @override
  final Tensor? bias;
  @override
  final SymmetricPadding2D stride;
  @override
  final SymmetricPadding2D padding;
  final SymmetricPadding2D outputPadding;
  @override
  final SymmetricPadding2D dilation;
  @override
  final int groups;

  Conv2DTranspose(
    this.weight, {
    super.name = 'conv',
    this.bias,
    this.groups = 1,
    this.stride = const SymmetricPadding2D.same(1),
    this.padding = const SymmetricPadding2D.same(0),
    this.outputPadding = const SymmetricPadding2D.same(0),
    this.dilation = const SymmetricPadding2D.same(1),
  }) : assert(groups > 0) {
    assert(numInChannels % groups == 0);
    assert(numOutChannels % groups == 0);
  }

  @override
  Tensor forward(Tensor input, {required Context context}) {
    context.onloadModule(this);
    return NN2DUtil.conv2dTranspose(
      input,
      weight,
      bias: bias,
      stride: stride,
      padding: padding,
      outputPadding: outputPadding,
      dilation: dilation,
      groups: groups,
    );
  }

  @override
  void resetParameters({Generator? generator}) {
    Init.kaimingUniform_(weight, a: sqrt(5), generator: generator);
    if (bias != null) {
      final fan = Init.calculateKaimingFan(weight);
      if (fan.fanIn != 0) {
        double bound = 1.0 / sqrt(fan.fanIn);
        bias!.uniform_(from: -bound, to: bound, generator: generator);
      }
    }
  }

  @override
  late final Iterable<Tensor> parameters = {weight, if (bias != null) bias!};

  @override
  final Iterable<Module> submodules = const [];

  /// For transposed convolution, weight shape is [inChannels, outChannels/groups, kH, kW]
  @override
  int get numInChannels => weight.shape[0] * groups;

  /// For transposed convolution, weight shape is [inChannels, outChannels/groups, kH, kW]
  @override
  int get numOutChannels => weight.shape[1] * groups;

  SymmetricPadding2D get kernelSize {
    final size = weight.shape;
    return SymmetricPadding2D(vertical: size[2], horizontal: size[3]);
  }

  @override
  late final Map<String, dynamic> meta = {
    "inChannel": numInChannels,
    "outChannel": numOutChannels,
    "kernelSize": kernelSize.to2List(),
    "stride": stride.to2List(),
    "padding": padding.to2List(),
    "outputPadding": outputPadding.to2List(),
    "dilation": dilation.to2List(),
    "groups": groups,
  };

  @override
  String toString() =>
      'Conv2DTransposed(${meta.entries.map((e) => '${e.key}: ${e.value}').join(', ')})';

  static Future<Conv2DTranspose> loadFromSafeTensor(
    SafeTensorLoader loader, {
    String prefix = '',
    String name = 'conv',
    int groups = 1,
    SymmetricPadding2D stride = const SymmetricPadding2D.same(1),
    SymmetricPadding2D padding = const SymmetricPadding2D.same(0),
    SymmetricPadding2D outputPadding = const SymmetricPadding2D.same(0),
    SymmetricPadding2D dilation = const SymmetricPadding2D.same(1),
  }) async {
    final weight = await loader.loadByName('${prefix}weight');
    Tensor? bias;
    if (loader.hasTensor('${prefix}bias')) {
      bias = await loader.loadByName('${prefix}bias');
    }

    return Conv2DTranspose(
      weight,
      name: name,
      bias: bias,
      groups: groups,
      padding: padding,
      stride: stride,
      outputPadding: outputPadding,
      dilation: dilation,
    );
  }

  static Conv2DTranspose make({
    String name = 'conv',
    required int numInChannels,
    required int numOutChannels,
    SymmetricPadding2D kernelSize = const SymmetricPadding2D.same(3),
    int groups = 1,
    SymmetricPadding2D stride = const SymmetricPadding2D.same(1),
    SymmetricPadding2D padding = const SymmetricPadding2D.same(0),
    SymmetricPadding2D outputPadding = const SymmetricPadding2D.same(0),
    SymmetricPadding2D dilation = const SymmetricPadding2D.same(1),
    bool hasBias = true,
    Generator? generator,
    DataType? dataType,
    Device? device,
  }) {
    // For transposed convolution, weight shape is [inChannels, outChannels/groups, kH, kW]
    Tensor weights = Tensor.empty(
      [
        numInChannels,
        numOutChannels ~/ groups,
        kernelSize.vertical,
        kernelSize.horizontal,
      ],
      datatype: dataType,
      device: device,
    );
    Tensor? bias;
    if (hasBias) {
      bias = Tensor.empty([numOutChannels], datatype: dataType, device: device);
    }
    return Conv2DTranspose(
      weights,
      name: name,
      bias: bias,
      groups: groups,
      stride: stride,
      padding: padding,
      outputPadding: outputPadding,
      dilation: dilation,
    )..resetParameters(generator: generator);
  }
}

enum PadMode {
  /// pads with a constant value, this value is specified with fill
  constant,
  reflect,
  replicate,
  circular;

  static PadMode? tryFromPytorchString(String? value) {
    if (value == null) return null;
    return fromPytorchString(value);
  }

  static PadMode fromPytorchString(String value) {
    switch (value) {
      case '':
      case 'constant':
      case 'zeros':
        return PadMode.constant;
      case 'reflect':
        return PadMode.reflect;
      case 'replicate':
        return PadMode.replicate;
      case 'circular':
        return PadMode.circular;
      default:
        throw UnimplementedError('Unknown pad mode: $value');
    }
  }
}

class Padding2D {
  final int left;
  final int right;
  final int top;
  final int bottom;

  const Padding2D({
    required this.left,
    required this.right,
    required this.top,
    required this.bottom,
  });

  List<int> to4List() => [left, right, top, bottom];

  (int, int, int, int) to4Tuple() => (left, right, top, bottom);

  (int, int) to2Tuple() => (left, top);
}

class SymmetricPadding2D implements Padding2D {
  final int vertical;
  final int horizontal;

  const SymmetricPadding2D({required this.vertical, required this.horizontal});

  const SymmetricPadding2D.same(int value)
    : vertical = value,
      horizontal = value;

  factory SymmetricPadding2D.fromPytorchString(String str) {
    {
      final ret = int.tryParse(str.trim());
      if (ret != null) return SymmetricPadding2D.same(ret);
    }
    if (str.startsWith('(') && str.endsWith(')')) {
      final parts = str.substring(1, str.length - 1).split(',');
      if (parts.length == 2) {
        return SymmetricPadding2D(
          vertical: int.parse(parts[0].trim()),
          horizontal: int.parse(parts[1].trim()),
        );
      }
    }
    throw UnimplementedError('Cannot parse SymmetricPadding2D from $str');
  }

  SymmetricPadding2D multiplySymmetric(SymmetricPadding2D other) {
    return SymmetricPadding2D(
      vertical: vertical * other.vertical,
      horizontal: horizontal * other.horizontal,
    );
  }

  SymmetricPadding2D subtractInt(int other) {
    return SymmetricPadding2D(
      vertical: vertical - other,
      horizontal: horizontal - other,
    );
  }

  SymmetricPadding2D divideInt(int other) {
    return SymmetricPadding2D(
      vertical: vertical ~/ other,
      horizontal: horizontal ~/ other,
    );
  }

  @override
  int get bottom => horizontal;

  @override
  int get left => vertical;

  @override
  int get right => vertical;

  @override
  int get top => horizontal;

  List<int> to2List() => [vertical, horizontal];

  @override
  List<int> to4List() => [horizontal, horizontal, vertical, vertical];

  @override
  (int, int, int, int) to4Tuple() =>
      (horizontal, horizontal, vertical, vertical);

  @override
  (int, int) to2Tuple() => (vertical, horizontal);

  @override
  bool operator ==(Object other) {
    if (other is SymmetricPadding2D) {
      return vertical == other.vertical && horizontal == other.horizontal;
    } else if (other is Padding2D) {
      return vertical == other.top &&
          vertical == other.bottom &&
          horizontal == other.left &&
          horizontal == other.right;
    } else {
      return false;
    }
  }

  @override
  int get hashCode => Object.hashAll([vertical, horizontal]);
}

class Conv2DPad {
  final PadMode padMode;
  final Padding2D padding;

  Conv2DPad({required this.padMode, required this.padding});
}
