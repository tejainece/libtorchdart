import 'package:tensor/tensor.dart';

void main() {
  print('Starting reproduction script...');
  final vocabSize = 14;
  final embedDim = 32;
  final embedding = EmbeddingLayer.make(
    numEmbeddings: vocabSize,
    embedDim: embedDim,
    name: 'emb',
  );
  print('Weights device: ${embedding.weights.device}');

  final context = Context.best();

  // Create indices within bounds
  final inputIds = Tensor.from(
    [0, 1, 2, 3, 4],
    [1, 5],
    datatype: DataType.int64,
  );

  print('Calling embedding forward...');
  final output = embedding.forward(inputIds, context: context);
  print('Embedding forward successful. Shape: ${output.shape}');
}
