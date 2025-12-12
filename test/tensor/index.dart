import 'package:tensor/tensor.dart';
import 'package:test/test.dart';

void main() {
  group('Temsor', () {
    setUp(() {
      // Additional setup goes here.
    });

    test('index', () {
      Tensor tensor = Tensor.arange(0, 70).view([2, 35]);
      tensor = tensor.index([.all, .to(14)]);
      print(tensor);
    });
  });
}
