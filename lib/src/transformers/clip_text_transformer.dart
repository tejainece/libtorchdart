import 'package:libtorchdart/libtorchdart.dart';
import 'package:libtorchdart/src/nn/embedding_layer.dart';

class ClipTextTransformer {
  // TODO
}

class ClipAttention {
  /*
    self_attn: ClipAttention,
    layer_norm1: nn::LayerNorm,
    mlp: ClipMlp,
    layer_norm2: nn::LayerNorm,
  */
}

class ClipEncoder {
  // TODO
}

class ClipTextEmbeddings {
  final EmbeddingLayer tokenEmbedding;
  final EmbeddingLayer positionEmbedding;
  final Tensor positionIds;

  ClipTextEmbeddings({
    required this.tokenEmbedding,
    required this.positionEmbedding,
    required this.positionIds,
  });

  static Future<ClipTextEmbeddings> loadPretrainedFromSafeTensor() async {
    // TODO
    throw UnimplementedError();
  }
}