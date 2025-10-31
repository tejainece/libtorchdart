import 'package:libtorchdart/libtorchdart.dart';

void main() {
  print(Tensor.ones([]));
  print(Tensor.ones([1]));
  print(Tensor.ones([7]));
  print(Tensor.ones([3, 4]));
  print(Tensor.ones([2, 3, 4]));
  print(Tensor.ones([1, 2, 3, 4]));
  print('Finished!');
} 
