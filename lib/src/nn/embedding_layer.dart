import 'dart:math';

import 'package:libtorchdart/libtorchdart.dart';

/*
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
*/

// TODO implement train/infer
abstract class Module {
  bool isTraining = false;

  void resetParameters();

  // TODO register parameter

  // Tensor forward(Tensor x);

  Map<String, dynamic> get meta;

  @override
  String toString() {
    return '$runtimeType(${meta.entries.map((e) => '${e.key}: ${e.value}').join(', ')})';
  }
}

abstract class SimpleModule implements Module {
  Tensor forward(Tensor x);
}

abstract class EmbeddableModule implements Module {
  Tensor forward(Tensor x, {Tensor? embeds});
}

abstract class InplaceModule implements Module {
  void forward_(Tensor x);
}

/// A simple lookup table that stores embeddings of a fixed dictionary and size.
///
/// This module is often used to retrieve word embeddings using indices.
/// The input to the module is a list of indices, and the embedding matrix,
/// and the output is the corresponding word embeddings.
class EmbeddingLayer extends Module implements SimpleModule {
  final Tensor weights;
  final bool sparse;
  final bool scaleGradByFreq;
  // TODO init
  final int? paddingIdx;
  final ({double maxNorm, double normType})? norm;

  EmbeddingLayer({
    required this.weights,
    required this.sparse,
    required this.scaleGradByFreq,
    required this.paddingIdx,
    required this.norm,
  }) {
    if (paddingIdx != null) {
      if (paddingIdx! > 0) {
        assert(paddingIdx! < numEmbeddings);
      } else if (paddingIdx! < 0) {
        assert(paddingIdx! >= -numEmbeddings);
      }
    }
  }

  @override
  Tensor forward(Tensor x) {
    return NNUtil.embedding(
      weights,
      x,
      paddingIdx: paddingIdx,
      scaleGradByFreq: scaleGradByFreq,
      sparse: sparse,
      norm: norm,
    );
  }

  @override
  void resetParameters() {
    weights.normal_();
    if (paddingIdx != null) {
      weights[paddingIdx!].fill_(0);
    }
  }

  int get numEmbeddings => weights.shape[0];

  int get embeddingDim => weights.shape[1];

  @override
  Map<String, dynamic> get meta => {
    "numEmbeddings": numEmbeddings,
    "embeddingDim": embeddingDim,
    "sparse": sparse,
    "scaleGradByFreq": scaleGradByFreq,
    "paddingIdx": paddingIdx,
    'maxNorm': norm?.maxNorm,
    'normType': norm?.normType,
  };

  static Future<EmbeddingLayer> loadFromSafeTensor(
    SafeTensorLoader loader, {
    String prefix = '',
    bool sparse = false,
    bool scaleGradByFreq = false,
    int? paddingIdx,
    ({double maxNorm, double normType})? norm,
  }) async {
    final weights = await loader.loadByName('${prefix}weight');
    return EmbeddingLayer(
      weights: weights,
      paddingIdx: paddingIdx,
      scaleGradByFreq: scaleGradByFreq,
      sparse: sparse,
      norm: norm,
    );
  }

  static EmbeddingLayer make(
    int numEmbeddings,
    int embeddingDim, {
    bool sparse = false,
    bool scaleGradByFreq = false,
    int? paddingIdx,
    ({double maxNorm, double normType})? norm,
  }) {
    return EmbeddingLayer(
      weights: Tensor.empty([numEmbeddings, embeddingDim]),
      paddingIdx: paddingIdx,
      scaleGradByFreq: scaleGradByFreq,
      sparse: sparse,
      norm: norm,
    );
  }
}

class Dropout extends Module implements SimpleModule, InplaceModule {
  final double p;

  Dropout(this.p) : assert(p >= 0 && p <= 1, 'p must be between 0 and 1');

  @override
  Tensor forward(Tensor x) {
    if (p == 0.0 || !isTraining) return x;
    return NNUtil.dropout(x, p, training: isTraining);
  }

  @override
  void forward_(Tensor x) {
    if (p == 0.0 || !isTraining) return;
    NNUtil.dropout_(x, p, training: isTraining);
  }

  @override
  void resetParameters() {}

  @override
  Map<String, dynamic> get meta => {"p": p};
}

class LinearLayer extends Module implements SimpleModule {
  final Tensor weight;
  final Tensor? bias;

  LinearLayer({required this.weight, this.bias});

  int get inFeatures => weight.shape[1];

  int get outFeatures => weight.shape[0];

  @override
  Tensor forward(Tensor x) {
    return NNUtil.linear(x, weight, bias: bias);
  }

  @override
  void resetParameters() {
    Init.kaimingUniform_(weight, a: sqrt(5));
    if (bias != null) {
      final fan = Init.calculateKaimingFan(weight);
      double bound = fan.fanIn > 0 ? 1 / sqrt(fan.fanIn) : 0;
      bias!.uniform_(from: -bound, to: bound);
    }
  }

  @override
  Map<String, dynamic> get meta => {
    "inFeatures": inFeatures,
    "outFeatures": outFeatures,
    "hasBias": bias != null,
  };

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

  static LinearLayer make({
    required int inFeatures,
    required int outFeatures,
    bool hasBias = true,
  }) {
    final weight = Tensor.empty([outFeatures, inFeatures]);
    final bias = hasBias ? Tensor.empty([outFeatures]) : null;
    return LinearLayer(weight: weight, bias: bias)..resetParameters();
  }
}
