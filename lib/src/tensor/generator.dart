import 'dart:ffi';

import 'package:ffi/ffi.dart';
import 'package:libtorchdart/libtorchdart.dart';
import 'package:libtorchdart/src/torch_ffi/torch_ffi.dart';

class Generator {
  final CGenerator nativePtr;

  Generator(this.nativePtr);

  set currentSeed(int seed) {
    FFIGenerator.setCurrentSeed(nativePtr, seed);
  }

  int get currentSeed => FFIGenerator.getCurrentSeed(nativePtr);

  Tensor get state {
    final state = FFIGenerator.getState(nativePtr);
    return Tensor(state);
  }

  set state(Tensor state) {
    FFIGenerator.setState(nativePtr, state.nativePtr);
  }

  // TODO set offset

  // TODO get offset

  static Generator getDefault({Device? device}) {
    final arena = Arena();
    try {
      Pointer<FFIDevice> devicePtr = nullptr;
      if (device != null) {
        devicePtr = FFIDevice.make(
          deviceType: device.deviceType,
          deviceIndex: device.deviceIndex,
          allocator: arena,
        );
      }
      final generator = FFIGenerator.getDefaultGeenrator(devicePtr);
      return Generator(generator);
    } finally {
      arena.releaseAll();
    }
  }
}
