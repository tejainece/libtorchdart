import 'dart:ffi';

import 'package:libtorchdart/src/torch_ffi/torch_ffi.dart';

typedef CGenerator = Pointer<Void>;

abstract class FFIGenerator {
  static final getDefaultGenerator = nativeLib
      .lookupFunction<
        CGenerator Function(Pointer<FFIDevice>),
        CGenerator Function(Pointer<FFIDevice>)
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
