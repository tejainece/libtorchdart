import 'package:tensor/tensor.dart';
import 'package:test/test.dart';

void main() {
  group('Tensor Squeeze/Unsqueeze', () {
    test('squeeze without dim', () {
      final tensor = Tensor.ones([1, 2, 1, 3, 1]);
      expect(tensor.shape, [1, 2, 1, 3, 1]);

      final squeezed = tensor.squeeze();
      expect(squeezed.shape, [2, 3]);
    });

    test('squeeze with dim', () {
      final tensor = Tensor.ones([1, 2, 1, 3, 1]);

      var squeezed = tensor.squeeze(dim: 0);
      expect(squeezed.shape, [2, 1, 3, 1]);

      squeezed = tensor.squeeze(dim: 2);
      expect(squeezed.shape, [1, 2, 3, 1]);

      // Squeeze dim 1 (size 2) - should not change
      squeezed = tensor.squeeze(dim: 1);
      expect(squeezed.shape, [1, 2, 1, 3, 1]);
    });

    test('unsqueeze', () {
      final tensor = Tensor.ones([2, 3]);
      expect(tensor.shape, [2, 3]);

      var unsqueezed = tensor.unsqueeze(0);
      expect(unsqueezed.shape, [1, 2, 3]);

      unsqueezed = tensor.unsqueeze(1);
      expect(unsqueezed.shape, [2, 1, 3]);

      unsqueezed = tensor.unsqueeze(2);
      expect(unsqueezed.shape, [2, 3, 1]);
    });
  });
}
