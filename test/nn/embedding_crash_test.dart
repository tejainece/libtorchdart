import 'package:test/test.dart';
import 'package:tensor/tensor.dart';

void main() {
  test('EmbeddingLayer crash reproduction', () {
    final vocabSize = 14;
    final embedDim = 32;
    final embedding = EmbeddingLayer.make(vocabSize, embedDim, name: 'emb');

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

    expect(output.shape, [1, 5, embedDim]);
  });
}
