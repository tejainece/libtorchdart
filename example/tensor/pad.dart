import 'package:libtorchdart/libtorchdart.dart';

void main() {
  final tensor = Tensor.ones([1, 2, 5, 7]);
  print(tensor);
  print('------------------------');
  print(tensor.pad([1, 2, 0, 0], mode: PadMode.circular));
}
