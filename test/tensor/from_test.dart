import 'package:tensor/tensor.dart';
import 'package:test/test.dart';

void main() {
  group('Tensor.from', () {
    test('Tensor.from creates valid tensor', () {
      final tensor = Tensor.from(
        [1.0, 2.0, 3.0],
        [3],
        datatype: DataType.float64,
      );

      expect(tensor.shape, [3]);
      expect(tensor.scalarAt(0), 1.0);
      expect(tensor.scalarAt(1), 2.0);
      expect(tensor.scalarAt(2), 3.0);
    });
  });
}
