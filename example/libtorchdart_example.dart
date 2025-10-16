import 'package:libtorchdart/libtorchdart.dart';

void main() {
  final tensor = Tensor.eye(7);
  print(tensor.dim);
  print(tensor.sizes);
  print(tensor.device);
  print(tensor);
}
