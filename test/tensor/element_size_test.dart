import 'package:libtorchdart/libtorchdart.dart';
import 'package:test/test.dart';

void main() {
  group('Tensor element_size', () {
    test('float32 element size', () {
      final tensor = Tensor.zeros([10], datatype: DataType.float32);
      expect(tensor.elementSize, 4);
    });

    test('float64 element size', () {
      final tensor = Tensor.zeros([10], datatype: DataType.float64);
      expect(tensor.elementSize, 8);
    });

    test('int32 element size', () {
      final tensor = Tensor.zeros([10], datatype: DataType.int32);
      expect(tensor.elementSize, 4);
    });

    test('int64 element size', () {
      final tensor = Tensor.zeros([10], datatype: DataType.int64);
      expect(tensor.elementSize, 8);
    });

    test('int8 element size', () {
      final tensor = Tensor.zeros([10], datatype: DataType.int8);
      expect(tensor.elementSize, 1);
    });

    test('uint8 element size', () {
      final tensor = Tensor.zeros([10], datatype: DataType.uint8);
      expect(tensor.elementSize, 1);
    });

    test('bool element size', () {
      final tensor = Tensor.zeros([10], datatype: DataType.boolean);
      expect(tensor.elementSize, 1);
    });
  });
}
