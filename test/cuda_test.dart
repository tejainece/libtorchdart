import 'package:tensor/tensor.dart';
import 'package:test/test.dart';

void main() {
  test('getDeviceProperties', () {
    if (!Device.isCudaAvailable) {
      print('CUDA not available, skipping getDeviceProperties test');
      return;
    }
    final props = Device.cuda().cudaDeviceProperties;
    expect(props, isNotNull);
    final name = props.name;
    expect(name, isNotEmpty);
    expect(props.totalMemory, greaterThan(0));
    expect(props.multiProcessorCount, greaterThan(0));
    expect(props.major, greaterThanOrEqualTo(0));
    expect(props.minor, greaterThanOrEqualTo(0));
  });

  test('memInfo', () {
    if (!Device.isCudaAvailable) {
      print('CUDA not available, skipping memInfo test');
      return;
    }
    final cuda = Device.cuda();
    // final tensor = Tensor.zeros([4], device: cuda);
    print(cuda.totalMemory);
    expect(cuda.totalMemory, greaterThan(0));
    print(cuda.freeMemory);
    expect(cuda.freeMemory, greaterThan(0));
    print(cuda.allocatedMemory);
    expect(cuda.allocatedMemory, isNot(throwsA(isA<Exception>())));
    expect(cuda.allocatedMemory, 0);
    print(cuda.reservedMemory);
    expect(cuda.reservedMemory, 0);

    final tensor = Tensor.zeros([7], device: cuda);

    expect(cuda.allocatedMemory, isNot(throwsA(isA<Exception>())));
    expect(cuda.allocatedMemory, greaterThan(0));
    print(cuda.reservedMemory);
    expect(cuda.reservedMemory, greaterThan(0));

    tensor.release();
    expect(cuda.allocatedMemory, 0);
    expect(cuda.reservedMemory, greaterThan(0));
  });
}
