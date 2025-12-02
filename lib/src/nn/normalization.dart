import 'package:collection/collection.dart';
import 'package:libtorchdart/libtorchdart.dart';

abstract class Normalization implements SimpleModule {}

abstract class EmbeddableNormalizer implements EmbeddableModule {}

class LayerNorm extends Module implements Normalization {
  final Tensor? weight;
  final Tensor? bias;
  final double eps;
  final List<int> normalizedShape;

  LayerNorm({
    super.name = 'norm',
    this.weight,
    this.bias,
    required this.normalizedShape,
    this.eps = 1e-5,
  }) {
    if (weight != null) {
      assert(weight!.dim == 1);
      assert(weight!.shape[0] == normalizedShape[0]);
      if (bias != null) {
        assert(bias!.dim == 1);
        assert(bias!.shape[0] == normalizedShape[0]);
      }
    }
  }

  @override
  Tensor forward(Tensor x, {Tensor? embeds, required Context context}) {
    context.onloadModule(this);
    return NNUtil.layerNorm(
      x,
      normalizedShape,
      weight: weight,
      bias: bias,
      eps: eps,
    );
  }

  @override
  void resetParameters() {
    weight?.ones_();
    bias?.zeros_();
  }

  @override
  late final Iterable<Tensor> parameters = [
    if (weight != null) weight!,
    if (bias != null) bias!,
  ];

  @override
  final Iterable<Module> submodules = const [];

  bool get isElementwiseAffine => weight != null && bias != null;

  bool get hasBias => bias != null;

  @override
  Map<String, dynamic> get meta => {
    "eps": eps,
    "isElementwiseAffine": isElementwiseAffine,
    "hasBias": hasBias,
    "normalizedShape": normalizedShape,
  };

  static Future<LayerNorm> loadFromSafeTensor(
    SafeTensorLoader loader, {
    String prefix = '',
    required String name,
    required List<int> normalizedShape,
    double eps = 1e-5,
  }) async {
    Tensor? weight;
    Tensor? bias;

    if (loader.hasTensor('${prefix}weight')) {
      weight = await loader.loadByName('${prefix}weight');
      if (loader.hasTensor('${prefix}bias')) {
        bias = await loader.loadByName('${prefix}bias');
      }
    }
    return LayerNorm(
      name: name,
      weight: weight,
      bias: bias,
      normalizedShape: normalizedShape,
      eps: eps,
    );
  }

  static LayerNorm make({
    required String name,
    required List<int> normalizedShape,
    double eps = 1e-5,
    bool isElementwiseAffine = true,
    bool hasBias = true,
    DataType? dataType,
    Device? device,
  }) {
    Tensor? weight;
    Tensor? bias;
    if (isElementwiseAffine) {
      weight = Tensor.ones(
        normalizedShape,
        datatype: dataType ?? DataType.float,
        device: device ?? Device.cpu,
      );
      if (hasBias) {
        bias = Tensor.zeros(
          normalizedShape,
          datatype: dataType ?? DataType.float,
          device: device ?? Device.cpu,
        );
      }
    }
    return LayerNorm(
      name: name,
      normalizedShape: normalizedShape,
      eps: eps,
      weight: weight,
      bias: bias,
    );
  }
}

/// Module that applies Group Normalization over mini-batch of inputs
/// as described by the paper https://arxiv.org/abs/1803.08494
class GroupNorm extends Module implements Normalization {
  final double eps;
  final Tensor? weight;
  final Tensor? bias;
  final int numGroups;

  GroupNorm({
    super.name = 'norm',
    required this.eps,
    required this.weight,
    required this.bias,
    required this.numGroups,
  }) {
    if (weight != null) {
      assert(weight!.dim == 1);
      assert(weight!.shape[0] % numGroups == 0);
    }
    if (bias != null) {
      assert(bias!.dim == 1);
    }
    if (weight != null && bias != null) {
      assert(weight!.shape[0] == bias!.shape[0]);
    }
  }

  @override
  Tensor forward(Tensor x, {required Context context}) {
    context.onloadModule(this);
    return NNUtil.groupNorm(x, numGroups, weight: weight, bias: bias, eps: eps);
  }

  @override
  void resetParameters() {
    weight?.ones_();
    bias?.zeros_();
  }

  @override
  late final Iterable<Tensor> parameters = [
    if (weight != null) weight!,
    if (bias != null) bias!,
  ];

  @override
  final Iterable<Module> submodules = const [];

  late final bool isElementwiseAffine = weight != null && bias != null;

  late final int? numChannels = weight?.shape[0];

  @override
  Map<String, dynamic> get meta => {
    "numGroups": numGroups,
    "numChannels": numChannels,
    "eps": eps,
    "isElementwiseAffine": isElementwiseAffine,
  };

  static Future<GroupNorm> loadFromSafeTensor(
    SafeTensorLoader loader, {
    String prefix = '',
    String name = 'norm',
    double eps = 1e-5,
    required int numGroups,
  }) async {
    Tensor? weight;
    Tensor? bias;

    if (loader.hasTensor('${prefix}weight')) {
      weight = await loader.loadByName('${prefix}weight');
    }
    if (loader.hasTensor('${prefix}bias')) {
      bias = await loader.loadByName('${prefix}bias');
    }

    return GroupNorm(
      name: name,
      eps: eps,
      weight: weight,
      bias: bias,
      numGroups: numGroups,
    );
  }

  static GroupNorm make({
    String name = 'norm',
    required int numGroups,
    required int numChannels,
    double eps = 1e-5,
    bool isElementwiseAffine = true,
    bool hasBias = true,
    DataType? dataType,
    Device? device,
  }) {
    assert(
      numChannels % numGroups == 0,
      'numChannels must be divisible by numGroups',
    );
    Tensor? weight;
    Tensor? bias;
    if (isElementwiseAffine) {
      weight = Tensor.empty(
        [numChannels],
        datatype: dataType ?? DataType.float,
        device: device ?? Device.cpu,
      );
      if (hasBias) {
        bias = Tensor.empty(
          [numChannels],
          datatype: dataType ?? DataType.float,
          device: device ?? Device.cpu,
        );
      }
    }
    return GroupNorm(
      name: name,
      eps: eps,
      weight: weight,
      bias: bias,
      numGroups: numGroups,
    )..resetParameters();
  }
}

/// Applies Root Mean Square Layer Normalization over a mini-batch of inputs.
class RMSNorm extends Module implements Normalization {
  final Tensor? weight;
  final List<int> normalizedShape;
  final double? eps;

  RMSNorm(this.normalizedShape, {super.name = 'norm', this.weight, this.eps});

  @override
  Tensor forward(Tensor x, {required Context context}) {
    context.onloadModule(this);
    return NNUtil.rmsNorm(x, normalizedShape, weight: weight, eps: eps);
  }

  @override
  Map<String, dynamic> get meta => {
    "eps": eps,
    "isElementwiseAffine": weight != null,
    "normalizedShape": normalizedShape,
  };

  @override
  void resetParameters() {
    weight?.ones_();
  }

  @override
  late final Iterable<Tensor> parameters = [if (weight != null) weight!];

  @override
  final Iterable<Module> submodules = const [];

  static Future<RMSNorm> loadFromSafeTensor(
    SafeTensorLoader loader, {
    String prefix = '',
    String name = 'norm',
    required List<int> normalizedShape,
    double? eps,
  }) async {
    Tensor? weight;
    if (loader.hasTensor('${prefix}weight')) {
      weight = await loader.loadByName('${prefix}weight');
      assert(const ListEquality().equals(weight.shape, normalizedShape));
    }
    return RMSNorm(normalizedShape, name: name, weight: weight, eps: eps);
  }

  static RMSNorm make({
    String name = 'norm',
    required List<int> normalizedShape,
    double? eps,
    bool isElementwiseAffine = true,
    DataType? dataType,
    Device? device,
  }) {
    Tensor? weight;
    if (isElementwiseAffine) {
      weight = Tensor.empty(
        normalizedShape,
        datatype: dataType ?? DataType.float,
        device: device ?? Device.cpu,
      );
    }
    return RMSNorm(name: name, normalizedShape, weight: weight, eps: eps)
      ..resetParameters();
  }
}

/// Root Mean Square layer normalization as described by https://huggingface.co/papers/1910.07467
class RMSNormWithBias extends Module implements Normalization {
  final Tensor? weight;
  final Tensor? bias;
  final double eps;

  RMSNormWithBias({
    super.name = 'norm',
    this.weight,
    this.bias,
    this.eps = 1e-5,
  });

  @override
  Tensor forward(Tensor x, {required Context context}) {
    context.onloadModule(this);
    Tensor variance = x.pow(2).mean(dim: [-1], keepDim: true);
    x = x * (variance + eps).rsqrt();

    if (weight != null) {
      x = x * weight!;
      if (bias != null) {
        x = x + bias!;
      }
    }

    return x;
  }

  @override
  void resetParameters() {
    weight?.ones_();
    bias?.zeros_();
  }

  @override
  late final Iterable<Tensor> parameters = [
    if (weight != null) weight!,
    if (bias != null) bias!,
  ];

  @override
  final Iterable<Module> submodules = const [];

  bool get isElementwiseAffine => weight != null || bias != null;

  @override
  Map<String, dynamic> get meta => {
    "eps": eps,
    "isElementwiseAffine": weight != null,
    "hasBias": bias != null,
  };

  static Future<RMSNormWithBias> loadFromSafeTensor(
    SafeTensorLoader loader, {
    String prefix = '',
    String name = 'norm',
    double eps = 1e-5,
  }) async {
    Tensor? weight;
    Tensor? bias;

    if (loader.hasTensor('${prefix}weight')) {
      weight = await loader.loadByName('${prefix}weight');
      if (loader.hasTensor('${prefix}bias')) {
        bias = await loader.loadByName('${prefix}bias');
      }
    }

    return RMSNormWithBias(name: name, weight: weight, bias: bias, eps: eps);
  }

  static RMSNormWithBias make({
    String name = 'norm',
    required List<int> normalizedShape,
    double eps = 1e-5,
    bool isElementwiseAffine = true,
    bool hasBias = true,
    DataType? dataType,
    Device? device,
  }) {
    Tensor? weight;
    Tensor? bias;
    if (isElementwiseAffine) {
      weight = Tensor.empty(
        normalizedShape,
        datatype: dataType ?? DataType.float,
        device: device ?? Device.cpu,
      );
      if (hasBias) {
        bias = Tensor.empty(
          normalizedShape,
          datatype: dataType ?? DataType.float,
          device: device ?? Device.cpu,
        );
      }
    }

    return RMSNormWithBias(name: name, weight: weight, bias: bias, eps: eps)
      ..resetParameters();
  }
}

/// Spatially conditioned normalization as defined in https://huggingface.co/papers/2209.09002
class SpatialNorm extends Module implements Normalization {
  final GroupNorm norm;
  final Conv2D convY;
  final Conv2D convB;

  SpatialNorm({
    super.name = 'norm',
    required this.norm,
    required this.convY,
    required this.convB,
  });

  @override
  Tensor forward(Tensor x, {required Context context}) {
    context.onloadModule(this);
    // TODO
    throw UnimplementedError();
  }

  @override
  void resetParameters() {
    // TODO
    throw UnimplementedError();
  }

  @override
  Map<String, dynamic> get meta => {
    // TODO
  };

  @override
  late final Iterable<Tensor> parameters = [
    // TODO
  ];

  @override
  late final Iterable<Module> submodules = [norm, convY, convB];
}
