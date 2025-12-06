import 'package:tensor/src/tensor/tensor.dart';

void main() {
  {
    final main = Tensor.arange(0, 10).reshape([5, 2]);
    print(main);

    final split = main.split([1, 2, 2]);
    print(split);

    final chunk = main.chunk(2);
    print(chunk);
  }

  {
    final main = Tensor.arange(0, 2).reshape([1, 2]);
    print(main);

    final chunk = main.chunk(2);
    print(chunk);
  }
}
