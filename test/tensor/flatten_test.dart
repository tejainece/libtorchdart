import 'package:test/test.dart';
import 'package:tensor/tensor.dart';

void main() {
  group('Tensor.flatten', () {
    test('flatten 3D tensor to 1D', () {
      final t = Tensor.from(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [2, 1, 3],
        datatype: DataType.float32,
      );

      final flattened = t.flatten();
      expect(flattened.shape, [6]);
      expect(flattened.at([0]).scalar, 1.0);
      expect(flattened.at([5]).scalar, 6.0);
    });

    test('flatten with startDim', () {
      final t = Tensor.randn([2, 3, 4], datatype: DataType.float32);
      // flatten from dim 1 to end: [2, 3*4] = [2, 12]
      final flattened = t.flatten(startDim: 1);
      expect(flattened.shape, [2, 12]);
    });

    test('flatten with endDim', () {
      final t = Tensor.randn([2, 3, 4, 5], datatype: DataType.float32);
      // flatten from 0 to 2: [2*3*4, 5] = [24, 5]
      final flattened = t.flatten(endDim: 2);
      expect(flattened.shape, [24, 5]);
    });

    test('flatten with startDim and endDim', () {
      final t = Tensor.randn([2, 3, 4, 5], datatype: DataType.float32);
      // flatten from 1 to 2: [2, 3*4, 5] = [2, 12, 5]
      final flattened = t.flatten(startDim: 1, endDim: 2);
      expect(flattened.shape, [2, 12, 5]);
    });
  });
}
