import 'dart:math' as math;
import 'package:libtorchdart/libtorchdart.dart';
import 'package:test/test.dart';

void main() {
  group('Tensor Mathematical Functions', () {
    test('sin function', () {
      final t1 = Tensor.from(
        [0.0, math.pi / 6, math.pi / 4, math.pi / 3, math.pi / 2],
        [5],
        datatype: DataType.float64,
      );
      final result = t1.sin();

      expect(result.shape, [5]);
      expect(result.scalarAt(0), closeTo(0.0, 1e-6));
      expect(result.scalarAt(1), closeTo(0.5, 1e-6));
      expect(result.scalarAt(2), closeTo(math.sqrt(2) / 2, 1e-6));
      expect(result.scalarAt(3), closeTo(math.sqrt(3) / 2, 1e-6));
      expect(result.scalarAt(4), closeTo(1.0, 1e-6));
    });

    test('cos function', () {
      final t1 = Tensor.from(
        [0.0, math.pi / 3, math.pi / 2, math.pi],
        [4],
        datatype: DataType.float64,
      );
      final result = t1.cos();

      expect(result.shape, [4]);
      expect(result.scalarAt(0), closeTo(1.0, 1e-6));
      expect(result.scalarAt(1), closeTo(0.5, 1e-6));
      expect(result.scalarAt(2), closeTo(0.0, 1e-6));
      expect(result.scalarAt(3), closeTo(-1.0, 1e-6));
    });

    test('exp function', () {
      final t1 = Tensor.from(
        [0.0, 1.0, 2.0, -1.0],
        [4],
        datatype: DataType.float64,
      );
      final result = t1.exp();

      expect(result.shape, [4]);
      expect(result.scalarAt(0), closeTo(1.0, 1e-6));
      expect(result.scalarAt(1), closeTo(math.e, 1e-6));
      expect(result.scalarAt(2), closeTo(math.e * math.e, 1e-6));
      expect(result.scalarAt(3), closeTo(1.0 / math.e, 1e-6));
    });

    test('rsqrt function', () {
      final t1 = Tensor.from(
        [1.0, 4.0, 9.0, 16.0],
        [4],
        datatype: DataType.float64,
      );
      final result = t1.rsqrt();

      expect(result.shape, [4]);
      expect(result.scalarAt(0), closeTo(1.0, 1e-6));
      expect(result.scalarAt(1), closeTo(0.5, 1e-6));
      expect(result.scalarAt(2), closeTo(1.0 / 3.0, 1e-6));
      expect(result.scalarAt(3), closeTo(0.25, 1e-6));
    });

    test('pow function with scalar exponent', () {
      final t1 = Tensor.from(
        [2.0, 3.0, 4.0, 5.0],
        [4],
        datatype: DataType.float64,
      );
      final result = t1.pow(2.0);

      expect(result.shape, [4]);
      expect(result.scalarAt(0), closeTo(4.0, 1e-6));
      expect(result.scalarAt(1), closeTo(9.0, 1e-6));
      expect(result.scalarAt(2), closeTo(16.0, 1e-6));
      expect(result.scalarAt(3), closeTo(25.0, 1e-6));
    });

    test('pow function with negative exponent', () {
      final t1 = Tensor.from([2.0, 4.0], [2], datatype: DataType.float64);
      final result = t1.pow(-1.0);

      expect(result.shape, [2]);
      expect(result.scalarAt(0), closeTo(0.5, 1e-6));
      expect(result.scalarAt(1), closeTo(0.25, 1e-6));
    });

    test('sin with multi-dimensional tensor', () {
      final t1 = Tensor.from(
        [0.0, math.pi / 2, math.pi, 3 * math.pi / 2],
        [2, 2],
        datatype: DataType.float64,
      );
      final result = t1.sin();

      expect(result.shape, [2, 2]);
      expect(result[0].scalarAt(0), closeTo(0.0, 1e-6));
      expect(result[0].scalarAt(1), closeTo(1.0, 1e-6));
      expect(result[1].scalarAt(0), closeTo(0.0, 1e-6));
      expect(result[1].scalarAt(1), closeTo(-1.0, 1e-6));
    });

    test('exp with multi-dimensional tensor', () {
      final t1 = Tensor.from(
        [0.0, 1.0, 2.0, 3.0],
        [2, 2],
        datatype: DataType.float64,
      );
      final result = t1.exp();

      expect(result.shape, [2, 2]);
      expect(result[0].scalarAt(0), closeTo(1.0, 1e-6));
      expect(result[0].scalarAt(1), closeTo(math.e, 1e-6));
    });

    test('chaining mathematical operations', () {
      final t1 = Tensor.from([1.0, 4.0, 9.0], [3], datatype: DataType.float64);
      // Test: sqrt(x) = 1/rsqrt(x), so x should equal (1/rsqrt(x))^2
      final rsqrtResult = t1.rsqrt();
      final squared = rsqrtResult.pow(-2.0);

      expect(squared.shape, [3]);
      expect(squared.scalarAt(0), closeTo(1.0, 1e-6));
      expect(squared.scalarAt(1), closeTo(4.0, 1e-6));
      expect(squared.scalarAt(2), closeTo(9.0, 1e-6));
    });
  });
}
