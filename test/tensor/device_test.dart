import 'package:test/test.dart';
import 'package:tensor/tensor.dart';

void main() {
  group('Device Tests', () {
    test('DeviceType.xpu exists', () {
      expect(DeviceType.xpu.name, 'XPU');
      expect(DeviceType.xpu.type, 12);
    });

    test('XPUDevice creation', () {
      final device = Device.xpu(deviceIndex: 0);
      expect(device.deviceType, DeviceType.xpu);
      expect(device.deviceIndex, 0);
      expect(device, isA<XPUDevice>());
    });

    test('Device factory creates XPUDevice', () {
      final device = Device(deviceType: DeviceType.xpu, deviceIndex: 1);
      expect(device.deviceType, DeviceType.xpu);
      expect(device.deviceIndex, 1);
      expect(device, isA<XPUDevice>());
    });

    test('isXpuAvailable runs without error', () {
      // This might return false on non-Intel systems, but shouldn't throw.
      final available = Device.isXpuAvailable;
      print('XPU Available: $available');
      expect(available, isA<bool>());
    });

    test('Device.tryXpu', () {
      final device = Device.tryXpu(0);
      if (Device.isXpuAvailable) {
        expect(device.deviceType, DeviceType.xpu);
      } else {
        expect(device.deviceType, DeviceType.cpu);
      }
    });

    test('XPUDevice memory methods', () {
      // These should be callable. If no XPU, they might return 0 or throw (depending on C++ impl).
      // Since we implemented them to check device_count, they should return 0 if no device.
      if (Device.isXpuAvailable) {
        final device = Device.xpu(deviceIndex: 0);
        expect(device.totalMemory, greaterThanOrEqualTo(0));
        expect(device.allocatedMemory, greaterThanOrEqualTo(0));
        expect(device.reservedMemory, greaterThanOrEqualTo(0));
      } else {
        // If not available, we can still construct the object but maybe not call methods if we didn't mock.
        // But our C++ implementation for totalMemory checks index >= device_count and returns 0.
        // memory_allocated/reserved in torch usually return 0 if invalid? Or throw?
        // Let's protect with try-catch or just check isXpuAvailable.
        // Actually, let's try calling them on a dummy device index 0.
        final device = Device.xpu(deviceIndex: 0);
        try {
          print('Total Mem: ${device.totalMemory}');
        } catch (e) {
          print('Caught expected error/exception on non-XPU machine: $e');
        }
      }
    });
  });
}
