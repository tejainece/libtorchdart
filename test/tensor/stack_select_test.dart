import 'package:tensor/tensor.dart';
import 'package:test/test.dart';

void main() {
  group('Tensor.stack', () {
    test('stack 1D tensors along dim 0', () {
      final a = Tensor.from([1.0, 2.0], [2], datatype: DataType.float32);
      final b = Tensor.from([3.0, 4.0], [2], datatype: DataType.float32);
      final c = Tensor.from([5.0, 6.0], [2], datatype: DataType.float32);

      Tensor stacked = Tensor.stack([a, b, c], dim: 0);

      expect(stacked.shape, [3, 2]);
      expect(stacked.index([0, 0]).scalar, 1.0);
      expect(stacked.index([0, 1]).scalar, 2.0);
      expect(stacked.index([1, 0]).scalar, 3.0);
      expect(stacked.index([1, 1]).scalar, 4.0);
      expect(stacked.index([2, 0]).scalar, 5.0);
      expect(stacked.index([2, 1]).scalar, 6.0);
    });

    test('stack 1D tensors along dim 1', () {
      final a = Tensor.from([1.0, 2.0], [2], datatype: DataType.float32);
      final b = Tensor.from([3.0, 4.0], [2], datatype: DataType.float32);

      final stacked = Tensor.stack([a, b], dim: 1);

      expect(stacked.shape, [2, 2]);
      expect(stacked.index([0, 0]).scalar, 1.0);
      expect(stacked.index([0, 1]).scalar, 3.0);
      expect(stacked.index([1, 0]).scalar, 2.0);
      expect(stacked.index([1, 1]).scalar, 4.0);
    });

    test('stack 2D tensors', () {
      final a = Tensor.from(
        [1.0, 2.0, 3.0, 4.0],
        [2, 2],
        datatype: DataType.float32,
      );
      final b = Tensor.from(
        [5.0, 6.0, 7.0, 8.0],
        [2, 2],
        datatype: DataType.float32,
      );

      final stacked = Tensor.stack([a, b], dim: 0);

      expect(stacked.shape, [2, 2, 2]);
      // First matrix
      expect(stacked.index([0, 0, 0]).scalar, 1.0);
      expect(stacked.index([0, 0, 1]).scalar, 2.0);
      expect(stacked.index([0, 1, 0]).scalar, 3.0);
      expect(stacked.index([0, 1, 1]).scalar, 4.0);
      // Second matrix
      expect(stacked.index([1, 0, 0]).scalar, 5.0);
      expect(stacked.index([1, 0, 1]).scalar, 6.0);
      expect(stacked.index([1, 1, 0]).scalar, 7.0);
      expect(stacked.index([1, 1, 1]).scalar, 8.0);
    });

    test('stack single tensor', () {
      final a = Tensor.from([1.0, 2.0, 3.0], [3], datatype: DataType.float32);
      final stacked = Tensor.stack([a], dim: 0);

      expect(stacked.shape, [1, 3]);
      expect(stacked.index([0, 0]).scalar, 1.0);
      expect(stacked.index([0, 1]).scalar, 2.0);
      expect(stacked.index([0, 2]).scalar, 3.0);
    });
  });

  group('Tensor.select', () {
    test('select from 2D tensor along dim 0', () {
      final t = Tensor.from(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [3, 2],
        datatype: DataType.float32,
      );

      final row0 = t.select(0, 0);
      expect(row0.shape, [2]);
      expect(row0.scalarAt(0), 1.0);
      expect(row0.scalarAt(1), 2.0);

      final row1 = t.select(0, 1);
      expect(row1.shape, [2]);
      expect(row1.scalarAt(0), 3.0);
      expect(row1.scalarAt(1), 4.0);

      final row2 = t.select(0, 2);
      expect(row2.shape, [2]);
      expect(row2.scalarAt(0), 5.0);
      expect(row2.scalarAt(1), 6.0);
    });

    test('select from 2D tensor along dim 1', () {
      final t = Tensor.from(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [3, 2],
        datatype: DataType.float32,
      );

      final col0 = t.select(1, 0);
      expect(col0.shape, [3]);
      expect(col0.scalarAt(0), 1.0);
      expect(col0.scalarAt(1), 3.0);
      expect(col0.scalarAt(2), 5.0);

      final col1 = t.select(1, 1);
      expect(col1.shape, [3]);
      expect(col1.scalarAt(0), 2.0);
      expect(col1.scalarAt(1), 4.0);
      expect(col1.scalarAt(2), 6.0);
    });

    test('select from 3D tensor', () {
      final t = Tensor.from(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        [2, 2, 2],
        datatype: DataType.float32,
      );

      final slice0 = t.select(0, 0);
      expect(slice0.shape, [2, 2]);
      expect(slice0[0][0].scalar, 1.0);
      expect(slice0[0][1].scalar, 2.0);
      expect(slice0[1][0].scalar, 3.0);
      expect(slice0[1][1].scalar, 4.0);

      final slice1 = t.select(0, 1);
      expect(slice1.shape, [2, 2]);
      expect(slice1[0][0].scalar, 5.0);
      expect(slice1[0][1].scalar, 6.0);
      expect(slice1[1][0].scalar, 7.0);
      expect(slice1[1][1].scalar, 8.0);
    });

    test('select is equivalent to indexing for dim 0', () {
      final t = Tensor.from(
        [1.0, 2.0, 3.0, 4.0],
        [2, 2],
        datatype: DataType.float32,
      );

      final selected = t.select(0, 1);
      final indexed = t[1];

      expect(selected.shape, indexed.shape);
      expect(selected.allClose(indexed), true);
    });
  });

  group('Tensor.stack and select integration', () {
    test('stack then select recovers original tensors', () {
      final a = Tensor.from([1.0, 2.0], [2], datatype: DataType.float32);
      final b = Tensor.from([3.0, 4.0], [2], datatype: DataType.float32);
      final c = Tensor.from([5.0, 6.0], [2], datatype: DataType.float32);

      final stacked = Tensor.stack([a, b, c], dim: 0);

      final recovered0 = stacked.select(0, 0);
      final recovered1 = stacked.select(0, 1);
      final recovered2 = stacked.select(0, 2);

      expect(recovered0.allClose(a), true);
      expect(recovered1.allClose(b), true);
      expect(recovered2.allClose(c), true);
    });
  });
}
