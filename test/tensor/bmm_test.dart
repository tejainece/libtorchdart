import 'package:tensor/tensor.dart';
import 'package:test/test.dart';

void main() {
  group('Tensor.bmm', () {
    test('simple batch matrix multiplication', () {
      // Batch size = 2, n = 2, m = 3, p = 2
      // input: (2, 2, 3)
      // mat2: (2, 3, 2)
      // output: (2, 2, 2)

      final input = Tensor.from(
        [
          // Batch 1
          1.0, 2.0, 3.0,
          4.0, 5.0, 6.0,
          // Batch 2
          1.0, 0.0, 0.0,
          0.0, 1.0, 0.0,
        ],
        [2, 2, 3],
        datatype: DataType.float32,
      );

      final mat2 = Tensor.from(
        [
          // Batch 1
          1.0, 2.0,
          3.0, 4.0,
          5.0, 6.0,
          // Batch 2
          10.0, 20.0,
          30.0, 40.0,
          50.0, 60.0,
        ],
        [2, 3, 2],
        datatype: DataType.float32,
      );

      final result = input.bmm(mat2);

      expect(result.shape, [2, 2, 2]);

      // Batch 1 calculation:
      // [1 2 3]   [1 2]   [1+6+15   2+8+18]   [22  28]
      // [4 5 6] x [3 4] = [4+15+30  8+20+36] = [49  64]
      //           [5 6]

      // Batch 2 calculation:
      // [1 0 0]   [10 20]   [10 20]
      // [0 1 0] x [30 40] = [30 40]
      //           [50 60]

      final expected = [22.0, 28.0, 49.0, 64.0, 10.0, 20.0, 30.0, 40.0];

      final resultList = result.toList();
      for (int i = 0; i < expected.length; i++) {
        expect(resultList[i], closeTo(expected[i], 1e-4));
      }
    });

    test('baddbmm with beta=0 matches bmm', () {
      final input = Tensor.randn([2, 5, 4]);
      final mat2 = Tensor.randn([2, 4, 3]);

      final resBmm = input.bmm(mat2);

      // baddbmm: out = beta * bias + alpha * (input @ mat2)
      // bias needs to be created with correct size
      final bias = Tensor.zeros([2, 5, 3]);
      final resBaddbmm = bias.baddbmm(input, mat2, beta: 0, alpha: 1);

      expect(resBmm.allClose(resBaddbmm), isTrue);
    });
  });
}
