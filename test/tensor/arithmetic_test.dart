import 'package:tensor/tensor.dart';
import 'package:test/test.dart';

void main() {
  group('Tensor Arithmetic', () {
    test('division by scalar', () {
      final t1 = Tensor.from(
        [2.0, 4.0, 6.0, 8.0],
        [4],
        datatype: DataType.float64,
      );
      final t2 = t1 / 2.0;
      expect(t2.shape, [4]);
      expect(t2.at([0]).scalar, 1.0);
      expect(t2.at([1]).scalar, 2.0);
      expect(t2.at([2]).scalar, 3.0);
      expect(t2.at([3]).scalar, 4.0);
    });
  });
}
