import 'dart:ffi';

import 'package:ffi/ffi.dart';
import 'package:libtorchdart/libtorchdart.dart';
import 'package:libtorchdart/src/torch_ffi/torch_ffi.dart';

typedef CGenerator = Pointer<Void>;

abstract class FFIGenerator {
  static final getDefaultGenerator = nativeLib
      .lookupFunction<
        CGenerator Function(Pointer<CDevice>),
        CGenerator Function(Pointer<CDevice>)
      >('torchffi_get_default_generator');

  static final getCurrentSeed = nativeLib
      .lookupFunction<Uint64 Function(CGenerator), int Function(CGenerator)>(
        'torchffi_generator_get_current_seed',
      );

  static final setCurrentSeed = nativeLib
      .lookupFunction<
        Void Function(CGenerator, Uint64),
        void Function(CGenerator, int)
      >('torchffi_generator_set_current_seed');

  static final getState = nativeLib
      .lookupFunction<
        CTensor Function(CGenerator),
        CTensor Function(CGenerator)
      >('torchffi_generator_get_state');

  static final setState = nativeLib
      .lookupFunction<
        Void Function(CGenerator, CTensor),
        void Function(CGenerator, CTensor)
      >('torchffi_generator_set_state');
}

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
      Pointer<CDevice> devicePtr = nullptr;
      if (device != null) {
        devicePtr = CDevice.make(
          deviceType: device.deviceType,
          deviceIndex: device.deviceIndex,
          allocator: arena,
        );
      }
      final generator = FFIGenerator.getDefaultGenerator(devicePtr);
      return Generator(generator);
    } finally {
      arena.releaseAll();
    }
  }
}
