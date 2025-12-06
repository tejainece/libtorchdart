import 'package:tensor/tensor.dart';

void main() {
  final generator = Generator.getDefault();
  print(generator.currentSeed);
  generator.currentSeed = 0;
  print(generator.currentSeed);
  print(Tensor.rand([1], generator: generator));
  print(generator.state);
}
