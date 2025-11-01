import 'package:libtorchdart/libtorchdart.dart';

abstract class Normalization {
  Tensor forward(Tensor x);
}

abstract class EmbeddableNormalizer implements Normalization {
  @override
  Tensor forward(Tensor hiddenStates, {Tensor? embeds});
}

class LayerNorm extends Module implements Normalization {
  final Tensor? weight;
  final Tensor? bias;
  final double eps;
  final List<int> normalizedShape;

  LayerNorm({
    this.weight,
    this.bias,
    required this.normalizedShape,
    this.eps = 1e-5,
  });

  bool get elementwiseAffine => weight != null && bias != null;

  bool get hasBias => bias != null;

  @override
  Tensor forward(Tensor x) {
    return x.layerNorm(normalizedShape, weight: weight, bias: bias, eps: eps);
  }

  static Future<LayerNorm> loadFromSafeTensor(
    SafeTensorLoader loader, {
    String prefix = '',
    required List<int> normalizedShape,
    double eps = 1e-5,
  }) async {
    Tensor? weight;
    Tensor? bias;

    if (loader.hasTensor('${prefix}weight')) {
      weight = await loader.loadByName('${prefix}weight');
    }
    if (loader.hasTensor('${prefix}bias')) {
      bias = await loader.loadByName('${prefix}bias');
    }
    return LayerNorm(
      weight: weight,
      bias: bias,
      normalizedShape: normalizedShape,
      eps: eps,
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

  late final bool isAffine = weight != null && bias != null;

  late final int? numChannels = weight?.shape[0];

  @override
  Tensor forward(Tensor x) {
    return x.groupNorm(numGroups, weight: weight, bias: bias, eps: eps);
  }

  static Future<GroupNorm> loadFromSafeTensor(
    SafeTensorLoader loader, {
    String prefix = '',
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
      eps: eps,
      weight: weight,
      bias: bias,
      numGroups: numGroups,
    );
  }
}

/// Spatially conditioned normalization as defined in https://huggingface.co/papers/2209.09002
class SpatialNorm extends Module implements Normalization {
  final GroupNorm norm;
  final Conv2D convY;
  final Conv2D convB;

  SpatialNorm({required this.norm, required this.convY, required this.convB});

  @override
  Tensor forward(Tensor x) {
    // TODO
    throw UnimplementedError();
  }
}
