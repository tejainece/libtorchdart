import 'dart:ffi';

import 'package:ffi/ffi.dart';
import 'package:libtorchdart/src/torch_ffi/torch_ffi.dart';

class DeviceType {
  final String name;
  final int type;

  const DeviceType(this.name, this.type);

  @override
  String toString() => name;

  static const cpu = DeviceType('CPU', 0);
  static const cuda = DeviceType('CUDA', 1);
  static const mkldnn = DeviceType('MKLDNN', 2);
  static const opengl = DeviceType('OpenGL', 3);
  static const opencl = DeviceType('OpenCL', 4);
  static const ideep = DeviceType('IDEEP', 5);
  static const hip = DeviceType('HIP', 6);
  static const fpga = DeviceType('FPGA', 7);

  /// ONNX Runtime / Microsoft
  static const maia = DeviceType('MAIA', 8);
  static const xla = DeviceType('XLA', 9);
  static const vulkan = DeviceType('Vulkan', 10);
  static const metal = DeviceType('Metal', 11);
  static const xpu = DeviceType('XPU', 12);
  static const mps = DeviceType('MPS', 13);

  /// Meta (tensors with no data)
  static const meta = DeviceType('Meta', 14);

  /// HPU / HABANA
  static const hpu = DeviceType('HPU', 15);
  // // SX-Aurora / NEC
  static const ve = DeviceType('VE', 16);

  /// Lazy Tensors
  static const lazy = DeviceType('Lazy', 17);

  /// Graphcore IPU
  static const ipu = DeviceType('IPU', 18);

  /// Meta training and inference devices
  static const mtia = DeviceType('MTIA', 19);

  static DeviceType fromId(int type) =>
      _byId[type] ?? DeviceType('Unknown', type);

  static final Map<int, DeviceType> _byId = Map.fromEntries(
    list.map((v) => MapEntry(v.type, v)),
  );

  static const List<DeviceType> list = [
    cpu,
    cuda,
    mkldnn,
    opengl,
    opencl,
    ideep,
    hip,
    fpga,
    maia,
    xla,
    vulkan,
    metal,
    xpu,
    mps,
    meta,
    hpu,
    ve,
    lazy,
    ipu,
    mtia,
  ];
}

abstract class Device {
  DeviceType get deviceType;
  int get deviceIndex;

  const Device.constant();

  factory Device({required DeviceType deviceType, required int deviceIndex}) {
    switch (deviceType) {
      case DeviceType.cpu:
        return cpu;
      case DeviceType.cuda:
        return CudaDevice(deviceIndex: deviceIndex);
      default:
        return UnknownDevice(deviceType: deviceType, deviceIndex: deviceIndex);
    }
  }

  static const cpu = CPUDevice();

  static CudaDevice cuda({int deviceIndex = -1}) =>
      CudaDevice(deviceIndex: deviceIndex);

  static Device tryCuda([int deviceIndex = -1]) {
    if (!isCudaAvailable) return cpu;
    return cuda(deviceIndex: deviceIndex);
  }

  static Device best() {
    if (isCudaAvailable) return cuda();
    // TODO check for metal
    // TODO check for mps
    return cpu;
  }

  int get totalMemory;

  int get allocatedMemory;

  int get reservedMemory;

  int get freeMemory => totalMemory - allocatedMemory;

  @override
  bool operator ==(Object other) {
    if (other is! Device) return false;
    return deviceType == other.deviceType && deviceIndex == other.deviceIndex;
  }

  @override
  String toString() => '$deviceType:$deviceIndex';

  static bool get isCudaAvailable => _FFIDevice.isCudaAvailable();

  @override
  int get hashCode => Object.hashAll([deviceType.type, deviceIndex]);
}

class CPUDevice extends Device {
  const CPUDevice() : super.constant();

  @override
  DeviceType get deviceType => DeviceType.cpu;

  @override
  int get deviceIndex => -1;

  @override
  int get totalMemory => throw UnimplementedError();

  @override
  int get allocatedMemory => throw UnimplementedError();

  @override
  int get reservedMemory => throw UnimplementedError();
}

class UnknownDevice extends Device {
  @override
  final DeviceType deviceType;
  @override
  final int deviceIndex;

  const UnknownDevice({required this.deviceType, required this.deviceIndex})
    : super.constant();

  @override
  int get totalMemory => throw UnimplementedError();

  @override
  int get allocatedMemory => throw UnimplementedError();

  @override
  int get reservedMemory => throw UnimplementedError();
}

class CudaDevice extends Device {
  @override
  DeviceType get deviceType => DeviceType.cuda;
  @override
  final int deviceIndex;

  const CudaDevice({this.deviceIndex = -1}) : super.constant();

  CudaDeviceProperties get cudaDeviceProperties {
    assert(deviceType == DeviceType.cuda);

    final cptr = _FFIDevice.getDeviceProperties(deviceIndex);
    try {
      return CudaDeviceProperties.fromPointer(cptr);
    } finally {
      malloc.free(cptr.ref.name);
      malloc.free(cptr);
    }
  }

  @override
  int get totalMemory => _FFIDevice.cudaMemoryTotal(deviceIndex);

  @override
  int get allocatedMemory {
    final errorPtr = malloc.allocate<Pointer<Utf8>>(sizeOf<Pointer<Utf8>>());
    try {
      errorPtr.value = nullptr;
      final ret = _FFIDevice.cudaMemoryAllocated(deviceIndex, errorPtr);
      if (errorPtr.value != nullptr) {
        final error = errorPtr.value.toDartString();
        throw Exception(error);
      }
      return ret;
    } finally {
      final dataPtr = errorPtr.value;
      if (dataPtr != nullptr) malloc.free(dataPtr);
      malloc.free(errorPtr);
    }
  }

  @override
  int get reservedMemory {
    final errorPtr = malloc.allocate<Pointer<Utf8>>(sizeOf<Pointer<Utf8>>());
    try {
      errorPtr.value = nullptr;
      final ret = _FFIDevice.cudaMemoryReserved(deviceIndex, errorPtr);
      if (errorPtr.value != nullptr) {
        final error = errorPtr.value.toDartString();
        throw Exception(error);
      }
      return ret;
    } finally {
      final dataPtr = errorPtr.value;
      if (dataPtr != nullptr) malloc.free(dataPtr);
      malloc.free(errorPtr);
    }
  }

  @override
  String toString() => '$deviceType:$deviceIndex';
}

final class CDevice extends Struct {
  @Int8()
  external int deviceType;
  @Int8()
  external int deviceIndex;

  static Pointer<CDevice> allocate(Allocator allocator) =>
      allocator.allocate<CDevice>(sizeOf<CDevice>());

  static Pointer<CDevice> make({
    required DeviceType deviceType,
    required int deviceIndex,
    required Allocator allocator,
  }) {
    final device = allocate(allocator);
    device.ref.deviceType = deviceType.type;
    device.ref.deviceIndex = deviceIndex;
    return device;
  }
}

final class CDeviceProperties extends Struct {
  external Pointer<Utf8> name;
  @Int64()
  external int totalMemory;
  @Int64()
  external int multiProcessorCount;
  @Int32()
  external int major;
  @Int32()
  external int minor;
}

class CudaDeviceProperties {
  final String name;
  final int totalMemory;
  final int multiProcessorCount;
  final int major;
  final int minor;

  CudaDeviceProperties({
    required this.name,
    required this.totalMemory,
    required this.multiProcessorCount,
    required this.major,
    required this.minor,
  });

  factory CudaDeviceProperties.fromPointer(Pointer<CDeviceProperties> pointer) {
    return CudaDeviceProperties(
      name: pointer.ref.name.toDartString(),
      totalMemory: pointer.ref.totalMemory,
      multiProcessorCount: pointer.ref.multiProcessorCount,
      major: pointer.ref.major,
      minor: pointer.ref.minor,
    );
  }
}

abstract class _FFIDevice {
  static final isCudaAvailable = nativeLib
      .lookupFunction<Bool Function(), bool Function()>(
        'torchffi_is_cuda_available',
      );

  static final getDeviceProperties = nativeLib
      .lookupFunction<
        Pointer<CDeviceProperties> Function(Int32),
        Pointer<CDeviceProperties> Function(int)
      >('torchffi_cuda_get_device_properties');

  static final cudaMemoryTotal = nativeLib
      .lookupFunction<Int64 Function(Int32), int Function(int)>(
        'torchffi_cuda_memory_total',
      );

  static final cudaMemoryAllocated = nativeLib
      .lookupFunction<
        Int64 Function(Int8, Pointer<Pointer<Utf8>>),
        int Function(int, Pointer<Pointer<Utf8>>)
      >('torchffi_cuda_memory_allocated');

  static final cudaMemoryReserved = nativeLib
      .lookupFunction<
        Int64 Function(Int8, Pointer<Pointer<Utf8>>),
        int Function(int, Pointer<Pointer<Utf8>>)
      >('torchffi_cuda_memory_reserved');
}
