import 'package:libtorchdart/libtorchdart.dart';
import 'package:libtorchdart/src/nn/nn2d.dart';

void main() {
  final tensor = Tensor.ones([1, 2, 5, 7]);
  print(tensor);
  print('------------------------');
  print(tensor.pad([1, 2, 0, 0], mode: PadMode.circular));
}
