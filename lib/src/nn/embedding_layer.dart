import 'package:libtorchdart/libtorchdart.dart';

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
}
