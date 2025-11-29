import 'dart:ffi';
import 'package:ffi/ffi.dart';

import 'package:libtorchdart/libtorchdart.dart';
import 'package:universal_io/io.dart';

export 'generator_ffi.dart';
export 'tensor_ffi.dart';

String getLibraryPath() {
  if (Platform.isMacOS) {
    return 'torchffi/src/build/libtorchffi.dylib';
  } else if (Platform.isLinux) {
    return 'torchffi/src/build/libtorchffi.so';
  } else if (Platform.isWindows) {
    return 'torchffi/src/build/libtorchffi.dll';
  } else {
    throw UnsupportedError('Unsupported platform: ${Platform.operatingSystem}');
  }
}

final DynamicLibrary nativeLib = DynamicLibrary.open(getLibraryPath());

final class FFIDevice extends Struct {
  @Int8()
  external int deviceType;
  @Int8()
  external int deviceIndex;

  static Pointer<FFIDevice> allocate(Allocator allocator) =>
      allocator.allocate<FFIDevice>(sizeOf<FFIDevice>());

  static Pointer<FFIDevice> make({
    required DeviceType deviceType,
    required int deviceIndex,
    required Allocator allocator,
  }) {
    final device = allocate(allocator);
    device.ref.deviceType = deviceType.type;
    device.ref.deviceIndex = deviceIndex;
    return device;
  }

  static final isCudaAvailable = nativeLib
      .lookupFunction<Bool Function(), bool Function()>(
        'torchffi_is_cuda_available',
      );
}

final class FFITensorOptions extends Struct {
  external Pointer<Int8> dataType;
  external Pointer<FFIDevice> device;
  external Pointer<Int8> layout;
  external Pointer<Int8> memoryFormat;
  external Pointer<Bool> requiresGrad;
  external Pointer<Bool> pinnedMemory;

  static Pointer<FFITensorOptions> allocate(Allocator allocator) =>
      allocator.allocate<FFITensorOptions>(sizeOf<FFITensorOptions>());

  static Pointer<FFITensorOptions> make({
    required DataType? dataType,
    required Device? device,
    required Layout? layout,
    required MemoryFormat? memoryFormat,
    required bool? requiresGrad,
    required bool? pinnedMemory,
    required Allocator allocator,
  }) {
    final options = allocate(allocator);
    if (dataType != null) {
      options.ref.dataType = allocator.allocate<Int8>(sizeOf<Int8>())
        ..value = dataType.type;
    }
    if (device != null) {
      options.ref.device = FFIDevice.make(
        deviceType: device.deviceType,
        deviceIndex: device.deviceIndex,
        allocator: allocator,
      );
    }
    if (layout != null) {
      options.ref.layout = allocator.allocate<Int8>(sizeOf<Int8>())
        ..value = layout.type;
    }
    if (memoryFormat != null) {
      options.ref.memoryFormat = allocator.allocate<Int8>(sizeOf<Int8>())
        ..value = memoryFormat.id;
    }
    if (requiresGrad != null) {
      options.ref.requiresGrad = allocator.allocate<Bool>(sizeOf<Bool>())
        ..value = requiresGrad;
    }
    if (pinnedMemory != null) {
      options.ref.pinnedMemory = allocator.allocate<Bool>(sizeOf<Bool>())
        ..value = pinnedMemory;
    }
    return options;
  }
}

final class _ScalarValue extends Union {
  @Bool()
  external bool b;

  @Int64()
  external int i;

  @Double()
  external double d;
}

final class FFIScalar extends Struct {
  @Int8()
  external int dataType;

  external _ScalarValue _value;

  void setBool(bool value) {
    dataType = 0;
    _value.b = value;
  }

  void setInt(int value) {
    dataType = 1;
    _value.i = value;
  }

  void setDouble(double value) {
    dataType = 2;
    _value.d = value;
  }

  void setValue(dynamic value) {
    if (value is bool) {
      setBool(value);
    } else if (value is int) {
      setInt(value);
    } else if (value is double) {
      setDouble(value);
    } else {
      throw ArgumentError('Unsupported scalar type: ${value.runtimeType}');
    }
  }

  dynamic get value {
    switch (dataType) {
      case 0:
        return _value.b;
      case 1:
        return _value.i;
      case 2:
        return _value.d;
      default:
        return null;
    }
  }

  static Pointer<FFIScalar> allocate(Allocator allocator) =>
      malloc.allocate<FFIScalar>(sizeOf<FFIScalar>());
}
