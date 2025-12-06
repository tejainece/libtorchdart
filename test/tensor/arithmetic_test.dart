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
      expect(t2.scalarAt(0), 1.0);
      expect(t2.scalarAt(1), 2.0);
      expect(t2.scalarAt(2), 3.0);
      expect(t2.scalarAt(3), 4.0);
    });
  });
}
