import 'package:tensor/tensor.dart';
import 'package:test/test.dart';

void main() {
  group('Tensor.tril', () {
    test('basic functionality', () {
      final t = Tensor.from(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        [3, 3],
        datatype: DataType.float32,
      );

      final lower = t.tril();

      // Expected:
      // 1 0 0
      // 4 5 0
      // 7 8 9

      expect(lower[0][0].scalar, equals(1.0));
      expect(lower[0][1].scalar, equals(0.0));
      expect(lower[0][2].scalar, equals(0.0));

      expect(lower[1][0].scalar, equals(4.0));
      expect(lower[1][1].scalar, equals(5.0));
      expect(lower[1][2].scalar, equals(0.0));

      expect(lower[2][0].scalar, equals(7.0));
      expect(lower[2][1].scalar, equals(8.0));
      expect(lower[2][2].scalar, equals(9.0));
    });

    test('with diagonal', () {
      final t = Tensor.ones([3, 3], dataType: DataType.float32);

      final lower = t.tril(diagonal: -1);

      // Expected (diagonal -1):
      // 0 0 0
      // 1 0 0
      // 1 1 0

      expect(lower[0][0].scalar, equals(0.0));
      expect(lower[1][0].scalar, equals(1.0));
      expect(lower[1][1].scalar, equals(0.0));
      expect(lower[2][1].scalar, equals(1.0));
    });
  });
}
