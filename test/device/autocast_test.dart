import 'package:tensor/tensor.dart';
import 'package:test/test.dart';

void main() {
  group('Autocast', () {
    test('Scope test', () async {
      final device = Device.best();

      expect(device.autocastEnabled, false);
      device.setAutocastEnabled(true);
      expect(device.autocastEnabled, true);
      device.setAutocastEnabled(false);
      expect(device.autocastEnabled, false);

      device.withAutocast(true, () {
        expect(device.autocastEnabled, true);
        device.withAutocast(false, () {
          expect(device.autocastEnabled, false);
        });
        expect(device.autocastEnabled, true);
      });
      expect(device.autocastEnabled, false);
    });

    test('Matmul autocast test', () {
      if (!Device.isCudaAvailable) {
        print('Skipping autocast op test because CUDA is not available.');
        return;
      }

      final device = Device.cuda();
      final a = Tensor.full(
        [16, 16],
        1.0,
        datatype: DataType.float32,
        device: device,
      );
      final b = Tensor.full(
        [16, 16],
        1.0,
        datatype: DataType.float32,
        device: device,
      );

      final cF32 = a.matmul(b);
      expect(cF32.dataType, DataType.float32);

      device.withAutocast(true, () {
        final cAuto = a.matmul(b);
        expect(cAuto.dataType, DataType.half);
      });
    });
  });
}
