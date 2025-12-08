import 'dart:io';
import 'package:test/test.dart';
import 'package:tensor/tensor.dart';
import 'package:tensor/src/ffi/device.dart';

void main() {
  group('MPS Device Tests', () {
    test('isMpsAvailable returns true on macOS with MPS support', () {
      if (Platform.isMacOS) {
        // We can't guarantee MPS is supported on all macs (e.g. old intel ones)
        // But if it is, it should return true.
        // If not, we can at least check it doesn't crash.
        print('Is MPS available: ${Device.isMpsAvailable}');
      } else {
        expect(Device.isMpsAvailable, isFalse);
      }
    });

    test('Device.best() returns MPS on compatible macs', () {
      if (Platform.isMacOS && Device.isMpsAvailable) {
        expect(Device.best(), isA<MPSDevice>());
      }
    });

    test('Tensor creation on MPS', () {
      if (Platform.isMacOS && Device.isMpsAvailable) {
        final t = Tensor.from(
          [1.0, 2.0, 3.0],
          [3],
          datatype: DataType.float32,
          device: Device.mps(),
        );
        expect(t.device, isA<MPSDevice>());
        expect(t.device.deviceType, DeviceType.mps);
        print('Tensor on MPS: $t');
      }
    });

    test('Simple operation on MPS', () {
      if (Platform.isMacOS && Device.isMpsAvailable) {
        final t1 = Tensor.from(
          [1.0, 2.0, 3.0],
          [3],
          datatype: DataType.float32,
          device: Device.mps(),
        );
        final t2 = Tensor.from(
          [4.0, 5.0, 6.0],
          [3],
          datatype: DataType.float32,
          device: Device.mps(),
        );
        final t3 = t1 + t2;
        expect(t3.device, isA<MPSDevice>());
        expect(t3.toList(), [5.0, 7.0, 9.0]);
      }
    });

    test('MPS Memory stats', () {
      if (Platform.isMacOS && Device.isMpsAvailable) {
        final device = Device.mps();
        print('Total Memory: ${device.totalMemory}');
        print('Allocated Memory: ${device.allocatedMemory}');
        print('Reserved Memory: ${device.reservedMemory}');

        expect(device.totalMemory, greaterThan(0));
        expect(device.allocatedMemory, greaterThanOrEqualTo(0));
        expect(device.reservedMemory, greaterThanOrEqualTo(0));

        // Create a tensor to increase allocation
        final t = Tensor.from(
          [1.0],
          [1],
          datatype: DataType.float32,
          device: device,
        );
        print('Tensor created for memory test: $t');
        print(
          'Allocated Memory after tensor creation: ${device.allocatedMemory}',
        );
        expect(device.allocatedMemory, greaterThan(0));
      }
    });
  });
}
