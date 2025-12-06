import 'package:tensor/tensor.dart';

void main() {
  final context = Context.best();

  final generator = Generator.getDefault(device: context.device);
  generator.currentSeed = 0;

  final conv = Conv2D.make(
    numInChannels: 32,
    numOutChannels: 32,
    padding: SymmetricPadding2D.same(1),
    stride: SymmetricPadding2D.same(1),
    generator: generator,
  );
  print(conv);
  // print(conv.weight);
  // print(conv.bias);

  final input = Tensor.ones([1, 32, 28, 28]);
  final output = conv.forward(input, context: context);
  print(output);
}
