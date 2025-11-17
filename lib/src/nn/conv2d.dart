import 'dart:math';

import 'package:libtorchdart/libtorchdart.dart';

class Conv2D extends Module implements SimpleModule {
  final Tensor weight;
  final Tensor? bias;
  final SymmetricPadding2D stride;

  /// Padding used with [padMode]
  final Conv2DPad? customPad;
  final SymmetricPadding2D? padding;
  final SymmetricPadding2D dilation;
  final int groups;

  Conv2D(
    this.weight, {
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
      bias: bias,
      groups: groups,
      stride: stride,
      customPad: customPad,
      padding: padding,
      dilation: dilation,
    );
  }

  @override
  Tensor forward(Tensor input) {
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

  int get numInChannels => weight.shape[1] * groups;

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
      bias: bias,
      groups: groups,
      stride: stride,
      padding: padding,
      dilation: dilation,
      padMode: padMode,
    )..resetParameters(generator: generator);
  }
}

enum PadMode {
  /// pads with a constant value, this value is specified with fill
  constant,
  reflect,
  replicate,
  circular,
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
