import 'dart:ffi';
import 'package:ffi/ffi.dart';
import 'package:tensor/src/ffi/tensor_ffi.dart';

import 'package:tensor/tensor.dart';
import 'package:universal_io/io.dart';
import 'device.dart';

export 'device.dart';
export 'generator_ffi.dart';
export 'tensor_ffi.dart';

String getLibraryPath() {
  if (Platform.isMacOS) {
    return 'torchffi/build/libtorchffi.dylib';
  } else if (Platform.isLinux) {
    return 'torchffi/build/libtorchffi.so';
  } else if (Platform.isWindows) {
    return 'torchffi/build/libtorchffi.dll';
  } else {
    throw UnsupportedError('Unsupported platform: ${Platform.operatingSystem}');
  }
}

final DynamicLibrary nativeLib = DynamicLibrary.open(getLibraryPath());

final class CTensorOptions extends Struct {
  external Pointer<Int8> dataType;
  external Pointer<CDevice> device;
  external Pointer<Int8> layout;
  external Pointer<Int8> memoryFormat;
  external Pointer<Bool> requiresGrad;
  external Pointer<Bool> pinnedMemory;

  static Pointer<CTensorOptions> allocate(Allocator allocator) =>
      allocator.allocate<CTensorOptions>(sizeOf<CTensorOptions>());

  static Pointer<CTensorOptions> make({
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
      options.ref.device = CDevice.make(
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

  external CTensor t;
}

final class CScalar extends Struct {
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

  void setTensor(Tensor value) {
    dataType = 70;
    _value.t = value.nativePtr;
  }

  void setValue(dynamic value) {
    if (value is bool) {
      setBool(value);
    } else if (value is int) {
      setInt(value);
    } else if (value is double) {
      setDouble(value);
    } else if (value is Tensor) {
      setTensor(value);
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

  static Pointer<CScalar> allocate(Allocator allocator) =>
      malloc.allocate<CScalar>(sizeOf<CScalar>());

  static Pointer<CScalar> allocateWithValue(
    Allocator allocator,
    dynamic value,
  ) => malloc.allocate<CScalar>(sizeOf<CScalar>())..ref.setValue(value);
}

final class CFInfo extends Struct {
  @Double()
  external double min;

  @Double()
  external double max;

  @Double()
  external double eps;

  @Double()
  external double tiny;

  @Double()
  external double resolution;

  static Pointer<CFInfo> allocate(Allocator allocator) =>
      malloc.allocate<CFInfo>(sizeOf<CFInfo>());
}

abstract class FFIFInfo {
  static final finfo = nativeLib
      .lookupFunction<
        Void Function(Int8 dtype, Pointer<CFInfo> info),
        void Function(int dtype, Pointer<CFInfo> info)
      >('torchffi_finfo');
}
