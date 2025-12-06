import 'package:tensor/tensor.dart';
import 'package:test/test.dart';

void main() {
  group('Temsor', () {
    setUp(() {
      // Additional setup goes here.
    });

    test('index', () {
      Tensor tensor = Tensor.arange(70).view([2, 35]);
      tensor = tensor.index([Slice(), Slice(end: 14)]);
      print(tensor);
    });
  });
}
