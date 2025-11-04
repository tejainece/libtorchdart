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

// TODO implement train/infer
abstract class Module {
  bool isTraining = false;

  void resetParameters();

  // TODO register parameter

  // Tensor forward(Tensor x);
}

abstract class SimpleModule implements Module {
  Tensor forward(Tensor x);
}

abstract class InplaceModule implements Module {
  Tensor forward_(Tensor x);
}

class EmbeddingLayer extends Module implements SimpleModule {
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

  @override
  void resetParameters() {
    // TODO
    throw UnimplementedError();
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

class Dropout extends Module implements SimpleModule {
  @override
  Tensor forward(Tensor x) {
    // TODO
    throw UnimplementedError();
  }

  @override
  void resetParameters() {
    // TODO
    throw UnimplementedError();
  }
}

class LinearLayer extends Module implements SimpleModule {
  final Tensor weight;
  final Tensor? bias;

  LinearLayer({required this.weight, this.bias});

  int get inFeatures => weight.shape[1];

  int get outFeatures => weight.shape[0];

  @override
  Tensor forward(Tensor x) {
    return linear(x, weight, bias: bias);
  }

  @override
  void resetParameters() {
    // TODO
    throw UnimplementedError();
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
