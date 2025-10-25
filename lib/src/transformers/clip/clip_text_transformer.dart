import 'package:libtorchdart/libtorchdart.dart';
import 'package:libtorchdart/src/nn/embedding_layer.dart';
import 'package:libtorchdart/src/safetensor/storage.dart';

class ClipTextTransformer implements Module {
  final ClipTextEmbeddings embeddings;
  final ClipEncoder encoder;
  final NormLayer norm;

  ClipTextTransformer({
    required this.embeddings,
    required this.encoder,
    required this.norm,
  });

  @override
  Tensor forward(Tensor inputIds) {
    final embeddings = this.embeddings.forward(inputIds);
    throw UnimplementedError();
  }

  static Future<ClipTextTransformer> loadPretrainedFromSafeTensor() async {
    // TODO
    throw UnimplementedError();
  }
}

class ClipEncoder {
  final List<ClipEncoderLayer> layers;

  ClipEncoder({required this.layers});

  Tensor forward(Tensor x, Tensor attentionMask) {
    // TODO
    throw UnimplementedError();
  }
}

class ClipEncoderLayer {
  Tensor forward(Tensor x, Tensor attentionMask) {
    throw UnimplementedError();
  }
}

class ClipTextEmbeddings implements Module {
  final EmbeddingLayer tokenEmbedding;
  final EmbeddingLayer positionEmbedding;
  final Tensor positionIds;

  ClipTextEmbeddings({
    required this.tokenEmbedding,
    required this.positionEmbedding,
    required this.positionIds,
  });

  @override
  Tensor forward(Tensor inputIds) {
    final seqLength = inputIds.shape.last;
    if (seqLength > maxPositionEmbeddings) {
      throw ArgumentError.value(
        'Input sequence length is greater than the maximum position embeddings',
      );
    }
    final positionEmbeddings = positionEmbedding.forward(positionIds);
    return tokenEmbedding.forward(inputIds) + positionEmbeddings;
  }

  int get embeddingDim => tokenEmbedding.embeddingDim;

  int get vocabSize => tokenEmbedding.numEmbeddings;

  int get maxPositionEmbeddings => positionEmbedding.numEmbeddings;

  static Future<ClipTextEmbeddings> loadFromSafeTensor(
    SafeTensorLoader loader, {
    String prefix = 'text_model.embeddings.',
  }) async {
    // final tokenEmbedding = loader.loadByName(prefix + 'token_embedding.weight');
    final tokenEmbedding = await EmbeddingLayer.loadFromSafeTensor(
      loader,
      prefix: '${prefix}token_embedding.',
    );
    final positionEmbedding = await EmbeddingLayer.loadFromSafeTensor(
      loader,
      prefix: '${prefix}position_embedding.',
    );
    final Tensor positionIds = Tensor.arange(positionEmbedding.numEmbeddings).expand([1, -1]);
    return ClipTextEmbeddings(
      tokenEmbedding: tokenEmbedding,
      positionEmbedding: positionEmbedding,
      positionIds: positionIds,
    );
  }
}
