import 'package:libtorchdart/libtorchdart.dart';
import 'package:libtorchdart/src/nn/embedding_layer.dart';
import 'package:libtorchdart/src/safetensor/storage.dart';

/// Multi-headed attention from 'Attention Is All You Need' paper
class ClipAttention {
  final LinearLayer kProj;
  final LinearLayer vProj;
  final LinearLayer qProj;
  final LinearLayer outProj;
  final int numAttensionHeads;

  /// Size of each attention head
  final int headDim;
  final double scale;
  final double dropout;
  final AttentionFunction attentionFunction;

  ClipAttention({
    required this.kProj,
    required this.vProj,
    required this.qProj,
    required this.outProj,
    required this.numAttensionHeads,
    required this.headDim,
    required this.scale,
    required this.dropout,
    required this.attentionFunction,
  });

  (Tensor, Tensor) forward(Tensor x, {Tensor? attentionMask}) {
    final [batchSize, seqLength, embedDim] = x.sizes;
    Tensor queries = qProj.forward(x);
    Tensor keys = kProj.forward(x);
    Tensor values = vProj.forward(x);

    queries = queries.view([batchSize, seqLength, -1, headDim]).transpose([
      1,
      2,
    ]);
    keys = keys.view([batchSize, seqLength, -1, headDim]).transpose([1, 2]);
    values = values.view([batchSize, seqLength, -1, headDim]).transpose([1, 2]);

    var (attentionOutput, attentionWeights) = attentionFunction.forward(
      queries,
      keys,
      values,
      attentionMask: attentionMask,
      scaling: scale,
      dropout: isTraining ? dropout : 0,
    );

    attentionOutput = attentionOutput.reshape([
      batchSize,
      seqLength,
      -1,
    ]).contiguous();
    attentionOutput = outProj.forward(attentionOutput);

    return (attentionOutput, attentionWeights);
  }

  int get embedDim => numAttensionHeads * headDim;

  static Future<ClipAttention> loadFromSafeTensor(
    SafeTensorLoader loader, {
    String prefix = '',
    required ClipTextConfig config,
  }) async {
    final qProj = await LinearLayer.loadFromSafeTensor(
      loader,
      prefix: '${prefix}q_proj.',
    );
    final kProj = await LinearLayer.loadFromSafeTensor(
      loader,
      prefix: '${prefix}k_proj.',
    );
    final vProj = await LinearLayer.loadFromSafeTensor(
      loader,
      prefix: '${prefix}v_proj.',
    );
    final outProj = await LinearLayer.loadFromSafeTensor(
      loader,
      prefix: '${prefix}out_proj.',
    );

    return ClipAttention(
      kProj: kProj,
      vProj: vProj,
      qProj: qProj,
      outProj: outProj,
      numAttensionHeads: 0, // TODO
      headDim: 0, // TODO
      scale: 0, // TODO
      dropout: config.attentionDropout,
      attentionFunction: EagerAttentionFunction(),
    );
  }
}

abstract class AttentionFunction {
  (Tensor, Tensor) forward(
    Tensor q,
    Tensor k,
    Tensor v, {
    Tensor? attentionMask,
    double dropout = 0,
    double? scaling,
  });
}

class EagerAttentionFunction implements AttentionFunction {
  @override
  (Tensor, Tensor) forward(
    Tensor q,
    Tensor k,
    Tensor v, {
    Tensor? attentionMask,
    double dropout = 0,
    double? scaling,
  }) {
    Tensor attentionWeights = q.matmul(k.transpose([-1, -2]));
    if (scaling != null) {
      attentionWeights = attentionWeights * scaling;
    }
    if (attentionMask != null) {
      attentionWeights = attentionWeights + attentionMask;
    }

    attentionWeights = attentionWeights
        .softmax(-1, dataType: DataType.float)
        .to(dataType: q.dataType);
    attentionWeights = attentionWeights.dropout(
      dropout,
      // TODO training: ,
    );

    Tensor output = attentionWeights.matmul(v);
    attentionWeights = attentionWeights.transpose([1, 2]).contiguous();

    return (output, attentionWeights);
  }
}
