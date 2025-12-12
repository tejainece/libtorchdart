import 'package:tensor/tensor.dart';
import 'package:test/test.dart';

void main() {
  group('Tensor Clone', () {
    test('clone tensor', () {
      final t1 = Tensor.ones([3], dataType: DataType.float64);
      final t2 = t1.clone();
      expect(t2.shape, [3]);
      expect(t2.toList(), [1.0, 1.0, 1.0]);

      t2.fill_(10.0);
      expect(t2.at([0]).scalar, 10.0);
      expect(t1.at([0]).scalar, 1.0);

      t1.fill_(20.0);
      expect(t1.at([0]).scalar, 20.0);
      expect(t2.at([0]).scalar, 10.0);
    });
  });
}
