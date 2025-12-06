import 'dart:async';
import 'dart:ffi';

import 'package:ffi/ffi.dart';
import 'package:tensor/tensor.dart';

abstract class SafeTensorLoader {
  SafeTensorHeader get header;

  Map<String, SafeTensorInfo> get tensorInfos => header.tensorInfos;

  bool hasTensor(String name) => tensorInfos.containsKey(name);

  bool hasTensorWithPrefix(String prefix) {
    for (final name in tensorInfos.keys) {
      if (name.startsWith(prefix)) return true;
    }
    return false;
  }

  FutureOr<Tensor> loadByName(String name, {Device device = Device.cpu});

  FutureOr<Tensor?> tryLoadByName(String name, {Device device = Device.cpu});
}

class FileIOSafeTensorLoader extends SafeTensorLoader {
  @override
  final SafeTensorHeader header;

  FileIOSafeTensorLoader({required this.header});

  @override
  Tensor loadByName(String name, {Device device = Device.cpu}) {
    // TODO
    throw UnimplementedError();
  }

  @override
  Tensor? tryLoadByName(String name, {Device device = Device.cpu}) {
    // TODO
    throw UnimplementedError();
  }
}

/// Loads tensor using mmap.
/// Achieves Zero copy tensor loading.
class MmapSafeTensorLoader extends SafeTensorLoader {
  @override
  final SafeTensorHeader header;
  final int fd;
  final int mmapedLength;
  final Pointer<Uint8> _pointer;

  MmapSafeTensorLoader._({
    required this.header,
    required this.fd,
    required this.mmapedLength,
    required Pointer<Uint8> pointer,
  }) : _pointer = pointer;

  @override
  Tensor? tryLoadByName(String name, {Device device = Device.cpu}) {
    final info = header.tensorInfos[name];
    if (info == null) return null;

    final datatype = DataType.fromSafeTensorName(info.dtype);
    if (datatype == null) {
      throw Exception('Unsupported safetensor datatype: ${info.dtype}');
    }

    final dataPointer = _pointer + header.dataOffset + info.startOffset;

    final tensor = Tensor.fromBlob(
      dataPointer.cast<Void>(),
      info.shape,
      datatype: datatype,
      device: Device.cpu,
      // TODO other parameters
    );

    if (device.deviceType != DeviceType.cpu) {
      return tensor.to(device: device);
    }
    return tensor;
  }

  @override
  Tensor loadByName(String name, {Device device = Device.cpu}) {
    final tensor = tryLoadByName(name, device: device);
    if (tensor == null) {
      throw Exception('Tensor $name not found');
    }
    return tensor;
  }

  void release() {
    munmap(_pointer, mmapedLength);
    close(fd);
  }

  static MmapSafeTensorLoader make({
    required SafeTensorHeader header,
    required String path,
    required int fileLength,
  }) {
    final fd = open(path.toNativeUtf8(), 0);
    if (fd == -1) {
      throw Exception('Failed to open file: $path');
    }
    final Pointer<Uint8> result = MmapSafeTensorLoader.mmap(
      nullptr,
      fileLength,
      1,
      2,
      fd,
      0,
    );
    if (result.address == -1) {
      throw Exception('mmap failed');
    }
    return MmapSafeTensorLoader._(
      header: header,
      fd: fd,
      mmapedLength: fileLength,
      pointer: result,
    );
  }

  @Native<Int Function(Pointer<Utf8>, Int)>(symbol: "open")
  external static int open(Pointer<Utf8> filename, int flags);

  @Native<Int Function(Int)>(symbol: "close")
  external static int close(int fd);

  @Native<IntPtr Function(Pointer<Uint8> address, Size len)>()
  external static int munmap(Pointer<Uint8> address, int len);

  @Native<
    Pointer<Uint8> Function(Pointer<Uint8>, Size, Int, Int, Int, IntPtr)
  >()
  external static Pointer<Uint8> mmap(
    Pointer<Uint8> address,
    int len,
    int prot,
    int flags,
    int fd,
    int offset,
  );
}

/// Loads tensor using Nvidia GPU Direct storage.
/// This can only be used for Nvidia cards that have GDS support.
/// This achieves true Zero copy tensor loading.
class CudaGDSSafeTensorLoader extends SafeTensorLoader {
  @override
  final SafeTensorHeader header;

  CudaGDSSafeTensorLoader({required this.header});

  @override
  Tensor loadByName(String name, {Device device = Device.cpu}) {
    // TODO
    throw UnimplementedError();
  }

  @override
  Tensor? tryLoadByName(String name, {Device device = Device.cpu}) {
    // TODO
    throw UnimplementedError();
  }
}
