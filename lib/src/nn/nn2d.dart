import 'package:libtorchdart/src/safetensor/storage.dart';
import 'package:libtorchdart/src/tensor/tensor.dart';

class Conv2D {
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
    this.stride = const SymmetricPadding2D(vertical: 1, horizontal: 1),
    this.customPad,
    this.padding,
    this.dilation = const SymmetricPadding2D(vertical: 1, horizontal: 1),
    this.groups = 1,
  }) : assert(padding != null || customPad != null);

  Tensor forward(Tensor input) {
    if (customPad == null) {
      return conv2d(
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
    return conv2d(
      input,
      weight,
      bias: bias,
      stride: stride,
      dilation: dilation,
      groups: groups,
    );
  }

  (int, int) get kernelSize {
    final size = weight.shape;
    return (size[2], size[3]);
  }

  static Future<Conv2D> loadFromSafeTensor(
    SafeTensorLoader loader, {
    String prefix = '',
    SymmetricPadding2D stride = const SymmetricPadding2D(
      vertical: 1,
      horizontal: 1,
    ),
    SymmetricPadding2D? padding,
    SymmetricPadding2D dilation = const SymmetricPadding2D(
      vertical: 1,
      horizontal: 1,
    ),
    int groups = 1,
    PadMode? padMode,
  }) async {
    // TODO transposed
    final weight = await loader.loadByName('${prefix}weight');
    Tensor? bias;
    if (loader.hasTensor('${prefix}bias')) {
      bias = await loader.loadByName('${prefix}bias');
    }

    Conv2DPad? customPad;
    if (padMode == null) {
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
      customPad = Conv2DPad(padMode: padMode!, padding: customPadding);
    }

    return Conv2D(
      weight,
      bias: bias,
      stride: stride,
      padding: padding,
      dilation: dilation,
      groups: groups,
      customPad: customPad,
    );
  }

  static Conv2D make({
    /// If true, uses padding so that the output size remains same as the input size
    bool padToSame = false,
  }) {
    throw UnimplementedError();
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

  @override
  List<int> to4List() => [horizontal, horizontal, vertical, vertical];

  @override
  (int, int, int, int) to4Tuple() =>
      (horizontal, horizontal, vertical, vertical);

  @override
  (int, int) to2Tuple() => (vertical, horizontal);
}

class Conv2DPad {
  final PadMode padMode;
  final Padding2D padding;

  Conv2DPad({required this.padMode, required this.padding});
}
