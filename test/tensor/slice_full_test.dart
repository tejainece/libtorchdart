import 'package:libtorchdart/libtorchdart.dart';
import 'package:test/test.dart';

void main() {
  group('Tensor.slice', () {
    test('slice 1D tensor', () {
      final t = Tensor.from(
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [5],
        datatype: DataType.float32,
      );

      final sliced = t.slice(0, 1, 4);
      expect(sliced.shape, [3]);
      expect(sliced.scalarAt(0), 2.0);
      expect(sliced.scalarAt(1), 3.0);
      expect(sliced.scalarAt(2), 4.0);
    });

    test('slice with step', () {
      final t = Tensor.from(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [6],
        datatype: DataType.float32,
      );

      final sliced = t.slice(0, 0, 6, step: 2);
      expect(sliced.shape, [3]);
      expect(sliced.scalarAt(0), 1.0);
      expect(sliced.scalarAt(1), 3.0);
      expect(sliced.scalarAt(2), 5.0);
    });

    test('slice to end with null', () {
      final t = Tensor.from(
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [5],
        datatype: DataType.float32,
      );

      final sliced = t.slice(0, 2, null);
      expect(sliced.shape, [3]);
      expect(sliced.scalarAt(0), 3.0);
      expect(sliced.scalarAt(1), 4.0);
      expect(sliced.scalarAt(2), 5.0);
    });

    test('slice 2D tensor along dim 0', () {
      final t = Tensor.from(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [3, 2],
        datatype: DataType.float32,
      );

      final sliced = t.slice(0, 1, 3);
      expect(sliced.shape, [2, 2]);
      expect(sliced.scalarAt(0), 3.0);
      expect(sliced.scalarAt(1), 4.0);
      expect(sliced.scalarAt(2), 5.0);
      expect(sliced.scalarAt(3), 6.0);
    });

    test('slice 2D tensor along dim 1', () {
      final t = Tensor.from(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [2, 3],
        datatype: DataType.float32,
      );

      final sliced = t.slice(1, 0, 2);
      expect(sliced.shape, [2, 2]);
      expect(sliced.scalarAt(0), 1.0);
      expect(sliced.scalarAt(1), 2.0);
      expect(sliced.scalarAt(2), 4.0);
      expect(sliced.scalarAt(3), 5.0);
    });
  });

  group('Tensor.full', () {
    test('create 1D tensor filled with value', () {
      final t = Tensor.full([5], 3.0, datatype: DataType.float32);

      expect(t.shape, [5]);
      for (int i = 0; i < 5; i++) {
        expect(t.scalarAt(i), 3.0);
      }
    });

    test('create 2D tensor filled with value', () {
      final t = Tensor.full([2, 3], 7.5, datatype: DataType.float32);

      expect(t.shape, [2, 3]);
      for (int i = 0; i < 6; i++) {
        expect(t.scalarAt(i), 7.5);
      }
    });

    test('create tensor filled with integer', () {
      final t = Tensor.full([3, 2], 42, datatype: DataType.int64);

      expect(t.shape, [3, 2]);
      for (int i = 0; i < 6; i++) {
        expect(t.scalarAt(i), 42);
      }
    });

    test('create tensor filled with zero', () {
      final t = Tensor.full([4], 0.0, datatype: DataType.float32);

      expect(t.shape, [4]);
      for (int i = 0; i < 4; i++) {
        expect(t.scalarAt(i), 0.0);
      }
    });

    test('create 3D tensor filled with value', () {
      final t = Tensor.full([2, 2, 2], 1.5, datatype: DataType.float32);

      expect(t.shape, [2, 2, 2]);
      for (int i = 0; i < 8; i++) {
        expect(t.scalarAt(i), 1.5);
      }
    });
  });

  group('Tensor.slice and full integration', () {
    test('slice a full tensor', () {
      final full = Tensor.full([10], 5.0, datatype: DataType.float32);
      final sliced = full.slice(0, 2, 7);

      expect(sliced.shape, [5]);
      for (int i = 0; i < 5; i++) {
        expect(sliced.scalarAt(i), 5.0);
      }
    });
  });
}
