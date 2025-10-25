import 'package:libtorchdart/src/nn/activation.dart';

class ClipConfig {
  final int vocabSize;
  final int intermediateSize;
  final int projectionDim;
  final int numHiddenLayers;
  final int numAttentionHeads;
  final int maxPositionEmbeddings;
  /// Embed dim/hidden size
  final int embedDim;
  final Activation activation;
  final String? padWith;

  const ClipConfig({
    required this.vocabSize,
    required this.intermediateSize,
    required this.projectionDim,
    required this.numHiddenLayers,
    required this.numAttentionHeads,
    required this.maxPositionEmbeddings,
    required this.embedDim,
    required this.activation,
    required this.padWith,
  });

  static const ClipConfig v1_5 = ClipConfig(
    vocabSize: 49408,
    embedDim: 768,
    activation: Activation.quickGelu,
    intermediateSize: 3072,
    maxPositionEmbeddings: 77,
    padWith: null,
    numHiddenLayers: 12,
    numAttentionHeads: 12,
    projectionDim: 768,
  );

  static const ClipConfig v2_1 = ClipConfig(
    vocabSize: 49408,
    embedDim: 1024,
    activation: Activation.gelu,
    intermediateSize: 4096,
    maxPositionEmbeddings: 77,
    padWith: "!",
    numHiddenLayers: 23,
    numAttentionHeads: 16,
    projectionDim: 512,
  );
}