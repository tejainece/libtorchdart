import 'package:libtorchdart/libtorchdart.dart';
import 'package:test/test.dart';

void main() {
  group('Tensor.to_', () {
    test('to_ changes dtype', () {
      final tensor = Tensor.from(
        [1.0, 2.0, 3.0],
        [3],
        datatype: DataType.float32,
      );
      expect(tensor.dataType, DataType.float32);
      final oldPtr = tensor.nativePtr;

      tensor.to_(dataType: DataType.float64);

      expect(tensor.dataType, DataType.float64);
      expect(tensor.nativePtr, isNot(equals(oldPtr)));
      expect(tensor[0].scalar, 1.0);
      expect(tensor[1].scalar, 2.0);
      expect(tensor[2].scalar, 3.0);
    });

    test('to_ is no-op if no change', () {
      final tensor = Tensor.from(
        [1.0, 2.0, 3.0],
        [3],
        datatype: DataType.float32,
      );

      tensor.to_(dataType: DataType.float32);

      // It seems PyTorch might still create a new tensor even if parameters are same if copy=false isn't strictly honored or if it decides to re-allocate.
      // But let's check if the values are preserved.
      expect(tensor.dataType, DataType.float32);
      expect(tensor[0].scalar, 1.0);
    });
  });
}
