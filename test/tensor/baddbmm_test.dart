import 'package:tensor/tensor.dart';
import 'package:test/test.dart';

void main() {
  group('baddbmm', () {
    test('baddbmm basic calculation', () {
      final input = Tensor.zeros([1, 2, 2]);
      final batch1 = Tensor.from(
        [1.0, 0.0, 0.0, 1.0],
        [1, 2, 2],
        datatype: DataType.float32,
      );
      final batch2 = Tensor.from(
        [2.0, 0.0, 0.0, 2.0],
        [1, 2, 2],
        datatype: DataType.float32,
      );

      // input + 1.0 * (batch1 @ batch2)
      // zeros + (identity * 2*identity) = 2*identity
      final result = input.baddbmm(batch1, batch2);

      expect(result.shape, [1, 2, 2]);
      final data = result.splitEqually(1, dim: 0)[0]; // get the first batch

      // Expected result is [[2, 0], [0, 2]]
      expect(result[0][0][0].scalar, 2.0);
      expect(result[0][1][1].scalar, 2.0);
    });

    test('baddbmm with alpha and beta', () {
      final input = Tensor.ones([1, 2, 2], dataType: DataType.float32);
      final batch1 = Tensor.eye(2).unsqueeze(0);
      final batch2 = Tensor.eye(2).unsqueeze(0);

      // result = beta * input + alpha * (batch1 @ batch2)
      // result = 0.5 * ones + 2.0 * (eye @ eye)
      // result = 0.5 + 2.0 * eye
      // diagonal = 2.5, off-diagonal = 0.5
      final result = input.baddbmm(batch1, batch2, beta: 0.5, alpha: 2.0);

      expect(result.shape, [1, 2, 2]);
      // diagonal
      expect(result.at([0, 0, 0]).scalar, closeTo(2.5, 1e-5));
      expect(result.at([0, 1, 1]).scalar, closeTo(2.5, 1e-5));
      // off-diagonal
      expect(result.at([0, 0, 1]).scalar, closeTo(0.5, 1e-5));
    });
  });
}
