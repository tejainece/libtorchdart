import 'package:tensor/tensor.dart';
import 'package:test/test.dart';
import 'dart:math' as math;

void main() {
  group('Tensor.norm', () {
    test('L2 norm of 1D tensor', () {
      final t = Tensor.from([3.0, 4.0], [2], datatype: DataType.float32);
      final norm = t.norm(2);

      expect(norm.shape, []);
      expect(norm.scalar, closeTo(5.0, 1e-6)); // sqrt(3^2 + 4^2) = 5
    });

    test('L1 norm of 1D tensor', () {
      final t = Tensor.from([3.0, -4.0], [2], datatype: DataType.float32);
      final norm = t.norm(1);

      expect(norm.shape, []);
      expect(norm.scalar, closeTo(7.0, 1e-6)); // |3| + |-4| = 7
    });

    test('L2 norm of 2D tensor', () {
      final t = Tensor.from(
        [1.0, 2.0, 3.0, 4.0],
        [2, 2],
        datatype: DataType.float32,
      );
      final norm = t.norm(2);

      // sqrt(1 + 4 + 9 + 16) = sqrt(30)
      expect(norm.shape, []);
      expect(norm.scalar, closeTo(math.sqrt(30), 1e-6));
    });

    test('norm along specific dimension', () {
      final t = Tensor.from(
        [1.0, 2.0, 3.0, 4.0],
        [2, 2],
        datatype: DataType.float32,
      );

      // Norm along dim 0 (columns)
      final colNorm = t.norm(2, dim: [0]);
      expect(colNorm.shape, [2]);
      expect(
        colNorm.scalarAt(0),
        closeTo(math.sqrt(1 + 9), 1e-6),
      ); // sqrt(1^2 + 3^2)
      expect(
        colNorm.scalarAt(1),
        closeTo(math.sqrt(4 + 16), 1e-6),
      ); // sqrt(2^2 + 4^2)
    });

    test('norm with keepDim', () {
      final t = Tensor.from(
        [1.0, 2.0, 3.0, 4.0],
        [2, 2],
        datatype: DataType.float32,
      );

      final norm = t.norm(2, dim: [0], keepDim: true);
      expect(norm.shape, [1, 2]);
    });

    test('infinity norm', () {
      final t = Tensor.from(
        [1.0, -5.0, 3.0, 2.0],
        [4],
        datatype: DataType.float32,
      );

      final norm = t.norm(double.infinity);
      expect(norm.shape, []);
      expect(norm.scalar, closeTo(5.0, 1e-6)); // max(|1|, |-5|, |3|, |2|) = 5
    });

    test('norm of 3D tensor along multiple dimensions', () {
      final t = Tensor.from(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        [2, 2, 2],
        datatype: DataType.float32,
      );

      final norm = t.norm(2, dim: [1, 2]);
      expect(norm.shape, [2]);

      // First slice: sqrt(1 + 4 + 9 + 16)
      expect(norm.scalarAt(0), closeTo(math.sqrt(30), 1e-6));
      // Second slice: sqrt(25 + 36 + 49 + 64)
      expect(norm.scalarAt(1), closeTo(math.sqrt(174), 1e-6));
    });

    test('L2 norm for normalization', () {
      final t = Tensor.from([3.0, 4.0], [2], datatype: DataType.float32);
      final normValue = t.norm(2);
      final normalized = t / normValue.scalar;

      // Normalized vector should have norm 1
      final normalizedNorm = normalized.norm(2);
      expect(normalizedNorm.scalar, closeTo(1.0, 1e-6));
    });

    test('norm with default p=2', () {
      final t = Tensor.from([1.0, 1.0, 1.0], [3], datatype: DataType.float32);
      final norm = t.norm(2);

      expect(norm.scalar, closeTo(math.sqrt(3), 1e-6));
    });
  });
}
