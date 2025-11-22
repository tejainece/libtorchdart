import 'package:libtorchdart/libtorchdart.dart';
import 'package:test/test.dart';

void main() {
  group('Tensor Clone', () {
    test('clone tensor', () {
      final t1 = Tensor.ones([3], datatype: DataType.float64);
      final t2 = t1.clone();
      expect(t2.shape, [3]);
      expect(t2.scalarAt(0), 1.0);
      expect(t2.scalarAt(1), 1.0);
      expect(t2.scalarAt(2), 1.0);

      t2.fill_(10.0);
      expect(t2.scalarAt(0), 10.0);
      expect(t1.scalarAt(0), 1.0);

      t1.fill_(20.0);
      expect(t1.scalarAt(0), 20.0);
      expect(t2.scalarAt(0), 10.0);
    });
  });
}
