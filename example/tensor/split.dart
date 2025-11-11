import 'package:libtorchdart/src/tensor/tensor.dart';

void main() {
  final main = Tensor.arange(10).reshape([5, 2]);
  print(main);
  final split = main.split([1, 2, 2]);
  print(split);
}
