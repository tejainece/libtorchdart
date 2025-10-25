import 'package:libtorchdart/libtorchdart.dart';
import 'package:libtorchdart/src/nn/embedding_layer.dart';

class ClipAttention {
  final LinearLayer k;
  final LinearLayer v;
  final LinearLayer q;
  final LinearLayer out;
  final int headDim;
  final double scale;
  final int numAttensionHeads;

  ClipAttention({
    required this.k,
    required this.v,
    required this.q,
    required this.out,
    required this.headDim,
    required this.scale,
    required this.numAttensionHeads,
  });

  Tensor forward(Tensor x, Tensor attentionMask) {
    // TODO
    throw UnimplementedError();
  }
}
