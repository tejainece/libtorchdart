import 'package:libtorchdart/libtorchdart.dart';
import 'package:libtorchdart/src/safetensor/storage.dart';

class EmbeddingConfig {
  final bool sparse;
  final bool scaleGradByFreq;
  // TODO init
  final int paddingIdx;

  const EmbeddingConfig({
    this.sparse = false,
    this.scaleGradByFreq = false,
    this.paddingIdx = -1,
  });
}

abstract class Module {
  bool isTraining = false;

  // TODO register parameter

  // Tensor forward(Tensor x);
}

abstract class SimpleModule {
  Tensor forward(Tensor x);
}

abstract class InplaceModule implements Module {
  Tensor forward_(Tensor x);
}

class EmbeddingLayer extends Module {
  final Tensor weights;
  final EmbeddingConfig config;

  EmbeddingLayer({required this.weights, required this.config});

  @override
  Tensor forward(Tensor x) {
    return embedding(
      weights,
      x,
      config.paddingIdx,
      config.scaleGradByFreq,
      config.sparse,
    );
  }

  static Future<EmbeddingLayer> loadFromSafeTensor(
    SafeTensorLoader loader, {
    String prefix = '',
  }) async {
    final weights = await loader.loadByName('${prefix}weight');
    return EmbeddingLayer(weights: weights, config: EmbeddingConfig());
  }

  int get numEmbeddings => weights.shape[0];

  int get embeddingDim => weights.shape[1];
}

class LayerNorm extends Module {
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
    return layerNorm(x, normalizedShape, weight, bias, eps);
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

class GroupNorm extends Module {
  @override
  Tensor forward(Tensor x) {
    // TODO
    throw UnimplementedError();
  }
}

class Dropout extends Module {
  @override
  Tensor forward(Tensor x) {
    // TODO
    throw UnimplementedError();
  }
}

class LinearLayer extends Module {
  final Tensor weight;
  final Tensor? bias;

  LinearLayer({required this.weight, this.bias});

  int get inFeatures => weight.shape[1];

  int get outFeatures => weight.shape[0];

  @override
  Tensor forward(Tensor x) {
    return linear(x, weight, bias: bias);
  }

  static Future<LinearLayer> loadFromSafeTensor(
    SafeTensorLoader loader, {
    String prefix = '',
  }) async {
    final weight = await loader.loadByName('${prefix}weight');
    Tensor? bias;
    if (loader.hasTensor('${prefix}bias')) {
      bias = await loader.loadByName('${prefix}bias');
    }
    return LinearLayer(weight: weight, bias: bias);
  }
}
