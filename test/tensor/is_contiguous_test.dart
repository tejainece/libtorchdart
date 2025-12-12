import 'package:tensor/tensor.dart';
import 'package:test/test.dart';

void main() {
  group('Tensor.isContiguous', () {
    test('should return true for a new tensor', () {
      final t = Tensor.from([1, 2, 3, 4], [2, 2], datatype: DataType.int32);
      expect(t.isContiguous(), isTrue);
    });

    test('should return false for a transposed tensor', () {
      final t = Tensor.from([1, 2, 3, 4], [2, 2], datatype: DataType.int32);
      final transposed = t.transpose(0, 1);
      expect(transposed.isContiguous(), isFalse);
    });

    test(
      'should return true after calling contiguous() on a non-contiguous tensor',
      () {
        final t = Tensor.from([1, 2, 3, 4], [2, 2], datatype: DataType.int32);
        final transposed = t.transpose(0, 1);
        final contiguous = transposed.contiguous();
        expect(contiguous.isContiguous(), isTrue);
      },
    );
  });
}
