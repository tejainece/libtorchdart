import 'package:libtorchdart/libtorchdart.dart';
import 'package:libtorchdart/src/diffusion/sd/stable_diffusion.dart';

class ClipTextTransformer extends Module implements TextEncoder {
  final ClipTextEmbeddings embeddings;
  final ClipEncoder encoder;
  final LayerNorm norm;

  ClipTextTransformer({
    required this.embeddings,
    required this.encoder,
    required this.norm,
  });

  Tensor forward(
    Tensor inputIds, {
    Tensor? attentionMask,
    Tensor? positionIds,
  }) {
    inputIds = inputIds.view([-1, inputIds.shape.last]);
    final hiddenStates = embeddings.forward(inputIds, positionIds: positionIds);
    final attentionMaskMade = createCausalMask(
      inputEmbeds: hiddenStates,
      attentionMask: attentionMask,
      cachePosition: Tensor.arange(
        hiddenStates.shape[1],
        device: hiddenStates.device,
      ),
    );
    Tensor lastHiddenState = encoder.forward(
      hiddenStates,
      attentionMask: attentionMaskMade,
    );
    lastHiddenState = norm.forward(lastHiddenState);

    /* TODO pooled layer
    Tensor pooledOutput = lastHiddenState.index([
      Tensor.arange(lastHiddenState.shape[0]),
      
    ]);*/

    return lastHiddenState;
  }

  @override
  void resetParameters() {
    // TODO
    throw UnimplementedError();
  }

  static Future<ClipTextTransformer> loadFromSafeTensor(
    SafeTensorLoader loader, {
    required ClipTextConfig config,
    String prefix = 'text_model.',
  }) async {
    final embeddings = await ClipTextEmbeddings.loadFromSafeTensor(
      loader,
      prefix: '${prefix}embeddings.',
    );
    final encoder = await ClipEncoder.loadFromSafeTensor(
      loader,
      prefix: '${prefix}encoder.',
      config: config,
    );
    final norm = await LayerNorm.loadFromSafeTensor(
      loader,
      prefix: '${prefix}final_layer_norm.',
      eps: config.layerNormEps,
      normalizedShape: [config.embedDim],
    );
    return ClipTextTransformer(
      embeddings: embeddings,
      encoder: encoder,
      norm: norm,
    );
  }
}

dynamic _processMaskArguments() {}

dynamic createCausalMask({
  required Tensor inputEmbeds,
  Tensor? attentionMask,
  required Tensor cachePosition,
  Tensor? positionIds,
}) {
  // TODO
}

class ClipEncoder {
  final List<ClipEncoderLayer> layers;

  ClipEncoder({required this.layers});

  Tensor forward(Tensor x, {Tensor? attentionMask}) {
    for (final layer in layers) {
      x = layer.forward(x, attentionMask: attentionMask);
    }
    return x;
  }

  static Future<ClipEncoder> loadFromSafeTensor(
    SafeTensorLoader loader, {
    String prefix = 'text_model.encoder.',
    required ClipTextConfig config,
  }) async {
    final layers = <ClipEncoderLayer>[];
    int layerId = 0;
    while (true) {
      if (!loader.hasTensor('${prefix}layer.$layerId')) {
        break;
      }
      final layer = await ClipEncoderLayer.loadFromSafeTensor(
        loader,
        prefix: '${prefix}layer.$layerId.',
        config: config,
      );
      layers.add(layer);
      layerId++;
    }
    return ClipEncoder(layers: layers);
  }
}

class ClipEncoderLayer {
  final ClipAttention selfAttention;
  final LayerNorm layerNorm1;
  final ClipMlp mlp;
  final LayerNorm layerNorm2;

  ClipEncoderLayer({
    required this.selfAttention,
    required this.layerNorm1,
    required this.mlp,
    required this.layerNorm2,
  });

  Tensor forward(Tensor x, {Tensor? attentionMask}) {
    Tensor residual = x;
    x = layerNorm1.forward(x);
    (x, _) = selfAttention.forward(x, attentionMask: attentionMask);
    x = residual + x;

    residual = x;
    x = layerNorm2.forward(x);
    x = mlp.forward(x);
    x = residual + x;
    return x;
  }

  static Future<ClipEncoderLayer> loadFromSafeTensor(
    SafeTensorLoader loader, {
    required ClipTextConfig config,

    String prefix = '',
  }) async {
    final selfAttention = await ClipAttention.loadFromSafeTensor(
      loader,
      prefix: '${prefix}self_attn.',
      config: config,
    );
    final layerNorm1 = await LayerNorm.loadFromSafeTensor(
      loader,
      prefix: '${prefix}layer_norm1.',
      normalizedShape: [config.embedDim],
      eps: config.layerNormEps,
    );
    final layerNorm2 = await LayerNorm.loadFromSafeTensor(
      loader,
      prefix: '${prefix}layer_norm2.',
      normalizedShape: [config.embedDim],
      eps: config.layerNormEps,
    );
    final mlp = await ClipMlp.loadFromSafeTensor(
      loader,
      prefix: '${prefix}mlp.',
      config: config,
    );
    return ClipEncoderLayer(
      selfAttention: selfAttention,
      layerNorm1: layerNorm1,
      mlp: mlp,
      layerNorm2: layerNorm2,
    );
  }
}

class ClipMlp extends Module implements SimpleModule {
  final LinearLayer linear1;
  final LinearLayer linear2;
  final Activation activation;

  ClipMlp({
    required this.linear1,
    required this.linear2,
    required this.activation,
  });

  @override
  Tensor forward(Tensor x) {
    x = linear1.forward(x);
    x = activation.forward(x);
    x = linear2.forward(x);
    return x;
  }

  @override
  void resetParameters() {
    // TODO
    throw UnimplementedError();
  }

  static Future<ClipMlp> loadFromSafeTensor(
    SafeTensorLoader loader, {
    required ClipTextConfig config,
    String prefix = '',
  }) async {
    final linear1 = await LinearLayer.loadFromSafeTensor(
      loader,
      prefix: '${prefix}fc1.',
    );
    final linear2 = await LinearLayer.loadFromSafeTensor(
      loader,
      prefix: '${prefix}fc2.',
    );
    // TODO verify shape against config
    return ClipMlp(
      linear1: linear1,
      linear2: linear2,
      activation: config.activation,
    );
  }
}

class ClipTextEmbeddings extends Module implements SimpleModule {
  final EmbeddingLayer tokenEmbedding;
  final EmbeddingLayer positionEmbedding;
  final Tensor positionIds;

  ClipTextEmbeddings({
    required this.tokenEmbedding,
    required this.positionEmbedding,
    required this.positionIds,
  });

  @override
  Tensor forward(Tensor inputIds, {Tensor? positionIds}) {
    final seqLength = inputIds.shape.last;
    if (seqLength > maxPositionEmbeddings) {
      throw ArgumentError.value(
        'Input sequence length is greater than the maximum position embeddings',
      );
    }
    positionIds ??= this.positionIds.expand([-1, seqLength]);
    final positionEmbeddings = positionEmbedding.forward(positionIds);
    return tokenEmbedding.forward(inputIds) + positionEmbeddings;
  }

  @override
  void resetParameters() {
    // TODO
    throw UnimplementedError();
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
    final Tensor positionIds = Tensor.arange(
      positionEmbedding.numEmbeddings,
    ).expand([1, -1]);
    return ClipTextEmbeddings(
      tokenEmbedding: tokenEmbedding,
      positionEmbedding: positionEmbedding,
      positionIds: positionIds,
    );
  }
}
