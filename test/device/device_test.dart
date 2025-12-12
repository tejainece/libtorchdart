import 'package:tensor/src/ffi/device.dart';
import 'package:test/test.dart';

void main() {
  group('Device Tests', () {
    test('Cuda device count', () {
      if (Device.isCudaAvailable) {
        final count = CudaDevice.deviceCount;
        print('CUDA Device Count: $count');
        expect(count, greaterThan(0));
      } else {
        final count = CudaDevice.deviceCount;
        print('CUDA Device Count (No CUDA): $count');
        expect(count, equals(0));
      }
    });

    // Also verify trying to access device properties
    test('Device properties', () {
      if (Device.isCudaAvailable) {
        final device = Device.cuda(deviceIndex: 0);
        print('Device: $device');
        print('Total Memory: ${device.totalMemory}');
      }
    });
    test('MPS device count', () {
      if (Device.isMpsAvailable) {
        final count = MPSDevice.deviceCount;
        print('MPS Device Count: $count');
        expect(count, greaterThan(0));
      } else {
        // Technically this might throw or handle gracefully depending on implementation,
        // but checking the property exists and is callable is good.
        // Given implementation: return torch::mps::is_available() ? 1 : 0;
        final count = MPSDevice.deviceCount;
        print('MPS Device Count (No MPS): $count');
        expect(count, equals(0));
      }
    });

    test('XPU device count', () {
      if (Device.isXpuAvailable) {
        final count = XPUDevice.deviceCount;
        print('XPU Device Count: $count');
        expect(count, greaterThan(0));
      } else {
        final count = XPUDevice.deviceCount;
        print('XPU Device Count (No XPU): $count');
        expect(count, equals(0));
      }
    });
  });
}
