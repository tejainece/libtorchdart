import 'package:test/test.dart';
import 'package:libtorchdart/libtorchdart.dart';

void main() {
  test('GPT2LMHeadModel forward pass', () {
    final config = GPT2Config(
      vocabSize: 100,
      nPositions: 20,
      nEmbd: 32,
      nLayer: 2,
      nHead: 4,
    );

    final model = GPT2LMHeadModel.make(config: config, name: 'gpt2');

    final batchSize = 2;
    final seqLength = 10;

    final inputIds = Tensor.ones([1, 10], datatype: DataType.int64);
    // final inputIds = (Tensor.rand([1, 10]) * config.vocabSize).to(dataType: DataType.int64);

    // Create dummy attention mask
    // final attentionMask = Tensor.ones([1, 10]);

    // Create dummy position ids
    final positionIds = Tensor.arange(
      10,
      datatype: DataType.int64,
    ).expand([1, 10]).unsqueeze(0).expand([batchSize, seqLength]);

    final context = Context.best();

    final output = model.forward(
      inputIds,
      positionIds: positionIds,
      context: context,
    );

    expect(output.shape, [batchSize, seqLength, config.vocabSize]);
  });
}
