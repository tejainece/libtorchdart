import 'dart:math';

import 'package:tensor/tensor.dart';

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
    required super.name,
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
  Tensor forward(Tensor x, {required Context context}) {
    context.onloadModule(this);
    // Ensure input is on the same device as the weights
    final inputs = x.to(device: context.device); // TODO remove if possible
    return NNUtil.embedding(
      weights,
      inputs,
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

  @override
  late final Iterable<Tensor> parameters = [weights];

  @override
  final Iterable<Module> submodules = const [];

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

  Future<void> loadFromSafeTensor(
    SafeTensorLoader loader, {
    String prefix = '',
  }) async {
    if (loader.hasTensor('${prefix}weight')) {
      final newWeights = await loader.loadByName('${prefix}weight');
      weights.copy_(newWeights);
    }
  }

  static Future<EmbeddingLayer> loadFromSafeTensorStatic(
    SafeTensorLoader loader, {
    String prefix = '',
    required String name,
    bool sparse = false,
    bool scaleGradByFreq = false,
    int? paddingIdx,
    ({double maxNorm, double normType})? norm,
  }) async {
    final weights = await loader.loadByName('${prefix}weight');
    return EmbeddingLayer(
      name: name,
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
    required String name,
    bool sparse = false,
    bool scaleGradByFreq = false,
    int? paddingIdx,
    ({double maxNorm, double normType})? norm,
  }) {
    return EmbeddingLayer(
      name: name,
      weights: Tensor.empty([numEmbeddings, embeddingDim]),
      paddingIdx: paddingIdx,
      scaleGradByFreq: scaleGradByFreq,
      sparse: sparse,
      norm: norm,
    )..resetParameters();
  }
}

class Dropout extends Module implements SimpleModule, InplaceModule {
  final double p;

  Dropout(this.p, {super.name = 'dropout'})
    : assert(p >= 0 && p <= 1, 'p must be between 0 and 1');

  @override
  Tensor forward(Tensor x, {required Context context}) {
    context.onloadModule(this);
    if (p == 0.0 || !context.isTraining) return x;
    return NNUtil.dropout(x, p, training: context.isTraining);
  }

  @override
  void forward_(Tensor x, {required Context context}) {
    if (p == 0.0 || !context.isTraining) return;
    NNUtil.dropout_(x, p, training: context.isTraining);
  }

  @override
  void resetParameters() {}

  @override
  Map<String, dynamic> get meta => {"p": p};

  @override
  late final Iterable<Tensor> parameters = const [];

  @override
  final Iterable<Module> submodules = const [];
}

class LinearLayer extends Module implements SimpleModule {
  final Tensor weight;
  final Tensor? bias;

  LinearLayer({super.name = 'linear', required this.weight, this.bias});

  int get inFeatures => weight.shape[1];

  int get outFeatures => weight.shape[0];

  @override
  Tensor forward(Tensor x, {required Context context}) {
    context.onloadModule(this);
    // Ensure input is on the same device as the weights
    final inputs = x.to(device: context.device); // TODO remove if possible
    return NNUtil.linear(inputs, weight, bias: bias);
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

  @override
  late final Iterable<Tensor> parameters = {weight, if (bias != null) bias!};

  @override
  final Iterable<Module> submodules = const [];

  Future<void> loadFromSafeTensor(
    SafeTensorLoader loader, {
    String prefix = '',
  }) async {
    if (loader.hasTensor('${prefix}weight')) {
      final newWeight = await loader.loadByName('${prefix}weight');
      weight.copy_(newWeight);
    }
    if (bias != null && loader.hasTensor('${prefix}bias')) {
      final newBias = await loader.loadByName('${prefix}bias');
      bias!.copy_(newBias);
    }
  }

  static Future<LinearLayer> loadFromSafeTensorStatic(
    SafeTensorLoader loader, {
    String prefix = '',
    String name = 'linear',
  }) async {
    final weight = await loader.loadByName('${prefix}weight');
    Tensor? bias;
    if (loader.hasTensor('${prefix}bias')) {
      bias = await loader.loadByName('${prefix}bias');
    }
    return LinearLayer(name: name, weight: weight, bias: bias);
  }

  static LinearLayer make({
    String name = 'linear',
    required int inFeatures,
    required int outFeatures,
    bool hasBias = true,
  }) {
    final weight = Tensor.empty([outFeatures, inFeatures]);
    final bias = hasBias ? Tensor.empty([outFeatures]) : null;
    return LinearLayer(name: name, weight: weight, bias: bias)
      ..resetParameters();
  }
}
