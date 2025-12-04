import 'package:libtorchdart/libtorchdart.dart';
import 'package:test/test.dart';

void main() {
  group('Tensor.randint', () {
    test('creates tensor with correct shape', () {
      final tensor = Tensor.randint(10, [2, 3]);
      expect(tensor.shape, [2, 3]);
      expect(tensor.numel, 6);
    });

    test('creates tensor with values in range [0, high)', () {
      final tensor = Tensor.randint(10, [100]);
      for (int i = 0; i < 100; i++) {
        final value = tensor.scalarAt(i);
        expect(value, greaterThanOrEqualTo(0));
        expect(value, lessThan(10));
      }
    });

    test('creates tensor with values in range [low, high)', () {
      final tensor = Tensor.randint(20, [100], low: 10);
      for (int i = 0; i < 100; i++) {
        final value = tensor.scalarAt(i);
        expect(value, greaterThanOrEqualTo(10));
        expect(value, lessThan(20));
      }
    });

    test('creates tensor with specified datatype', () {
      final tensor = Tensor.randint(10, [2, 3], datatype: DataType.int64);
      expect(tensor.dataType, DataType.int64);
    });

    test('creates tensor with specified device', () {
      final tensor = Tensor.randint(10, [2, 3], device: Device.cpu);
      expect(tensor.device.deviceType, DeviceType.cpu);
    });

    test('creates tensor with generator', () {
      final generator = Generator.getDefault();
      generator.currentSeed = 42;
      final tensor1 = Tensor.randint(100, [5], generator: generator);

      generator.currentSeed = 42;
      final tensor2 = Tensor.randint(100, [5], generator: generator);

      // With same seed, should produce same values
      for (int i = 0; i < 5; i++) {
        expect(tensor1.scalarAt(i), tensor2.scalarAt(i));
      }
    });

    test('creates 1D tensor', () {
      final tensor = Tensor.randint(5, [10]);
      expect(tensor.dim, 1);
      expect(tensor.shape, [10]);
    });

    test('creates 2D tensor', () {
      final tensor = Tensor.randint(5, [3, 4]);
      expect(tensor.dim, 2);
      expect(tensor.shape, [3, 4]);
    });

    test('creates 3D tensor', () {
      final tensor = Tensor.randint(5, [2, 3, 4]);
      expect(tensor.dim, 3);
      expect(tensor.shape, [2, 3, 4]);
    });
  });
}
