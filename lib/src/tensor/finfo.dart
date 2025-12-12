import 'dart:ffi';

import 'package:tensor/src/ffi/torch_ffi.dart';
import 'package:tensor/tensor.dart';

/// Machine limits for floating point types.
///
/// This class provides properties similar to `torch.finfo` in PyTorch.
class FInfo {
  final double min;
  final double max;
  final double eps;
  final double tiny;
  final double resolution;
  final DataType dtype;

  const FInfo._({
    required this.min,
    required this.max,
    required this.eps,
    required this.tiny,
    required this.resolution,
    required this.dtype,
  });

  factory FInfo.fromCFInfo(Pointer<CFInfo> info, DataType dtype) {
    return FInfo._(
      min: info.ref.min,
      max: info.ref.max,
      eps: info.ref.eps,
      tiny: info.ref.tiny,
      resolution: info.ref.resolution,
      dtype: dtype,
    );
  }
}
