import 'package:tensor/tensor.dart';
import 'package:test/test.dart';

void main() {
  group('Tensor.arange', () {
    test('arange(end)', () {
      final t = Tensor.arange(0, 5.0);
      expect(t.shape, [5]);
      expect(t.toList(), [0.0, 1.0, 2.0, 3.0, 4.0]);
    });

    test('arange(start, end)', () {
      final t = Tensor.arange(2, 7);
      expect(t.shape, [5]);
      expect(t.toList(), [2.0, 3.0, 4.0, 5.0, 6.0]);
    });

    test('arange(start, end, step)', () {
      final t = Tensor.arange(0, 10, step: 2);
      expect(t.shape, [5]);
      expect(t.toList(), [0.0, 2.0, 4.0, 6.0, 8.0]);
    });

    test('arange with float step', () {
      final t = Tensor.arange(0, 1, step: 0.2);
      expect(t.shape, [5]);
      // Precision might slightly vary, checking roughly
      final data = t.toList();
      expect(data[0], closeTo(0.0, 1e-6));
      expect(data[1], closeTo(0.2, 1e-6));
      expect(data[2], closeTo(0.4, 1e-6));
      expect(data[3], closeTo(0.6, 1e-6));
      expect(data[4], closeTo(0.8, 1e-6));
    });

    test('arange with float start/end', () {
      final t = Tensor.arange(0.5, 2.5);
      expect(t.shape, [2]);
      expect(t.toList(), [0.5, 1.5]);
    });
  });
}
