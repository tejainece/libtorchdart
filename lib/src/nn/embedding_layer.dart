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
  Tensor forward(Tensor x);
}

class EmbeddingLayer implements Module {
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

class NormLayer implements Module {
  @override
  Tensor forward(Tensor x) {
    throw UnimplementedError();
  }
}

class LinearLayer implements Module {
  @override
  Tensor forward(Tensor x) {
    throw UnimplementedError();
  }
}
