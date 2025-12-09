import 'package:tensor/src/tensor/tensor.dart';

void main() {
  final tensor = Tensor.randn([1, 3, 1]);
  print(tensor.expand([1, 1, 2, -1, 7]));

  print(tensor.repeat([1, 6, 1]));

  {
    final tensor = Tensor.randn([3]);
    print(tensor.view([1, 3, 1]));
  }
}
