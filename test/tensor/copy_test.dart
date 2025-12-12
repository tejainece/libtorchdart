import 'package:tensor/tensor.dart';
import 'package:test/test.dart';

void main() {
  group('Tensor.copy_', () {
    test('copy_ from same shape and type', () {
      final t1 = Tensor.zeros([2, 2], datatype: DataType.float32);
      final t2 = Tensor.ones([2, 2], dataType: DataType.float32);

      t1.copy_(t2);

      expect(t1.allClose(t2), isTrue);
    });

    test('copy_ with broadcasting', () {
      final t1 = Tensor.zeros([2, 2], datatype: DataType.float32);
      final t2 = Tensor.from([1.0], [1], datatype: DataType.float32);

      t1.copy_(t2);

      final expected = Tensor.ones([2, 2], dataType: DataType.float32);
      expect(t1.allClose(expected), isTrue);
    });

    test('copy_ with type casting', () {
      final t1 = Tensor.zeros([2], datatype: DataType.float32);
      final t2 = Tensor.from([1, 2], [2], datatype: DataType.int32);

      t1.copy_(t2);

      final expected = Tensor.from([1.0, 2.0], [2], datatype: DataType.float32);
      expect(t1.allClose(expected), isTrue);
    });
  });
}
