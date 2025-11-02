import 'dart:math';

import 'package:libtorchdart/libtorchdart.dart';

/// A 2D upsampling layer with an optional convolution.
class Upsample2D extends Module {
  final Normalization? norm;
  final Conv2D conv;
  final bool useConvTransposed;
  final bool interpolate;

  Upsample2D({
    required this.norm,
    required this.conv,
    this.useConvTransposed = false,
    this.interpolate = true,
  });

  Tensor forward(Tensor hiddenStates, {SymmetricPadding2D? outputSize}) {
    if (norm != null) {
      hiddenStates = norm!.forward(hiddenStates.permute([0, 2, 3, 1])).permute([
        0,
        3,
        1,
        2,
      ]);
    }

    if (useConvTransposed) {
      return conv.forward(hiddenStates);
    }

    if (hiddenStates.shape[0] >= 64) {
      hiddenStates = hiddenStates.contiguous();
    }

    if (interpolate) {
      if (outputSize != null) {
        final scale = [
          outputSize.vertical / hiddenStates.shape[2],
          outputSize.horizontal / hiddenStates.shape[3],
        ].reduce(max);

        if (hiddenStates.numel * scale > 2 >> 31) {
          hiddenStates = hiddenStates.contiguous();
        }

        hiddenStates = interpolateNearest(hiddenStates, outputSize.to2List());
      } else {
        if (hiddenStates.numel * 2 > 2 >> 31) {
          hiddenStates = hiddenStates.contiguous();
        }

        hiddenStates = interpolateNearestScale(hiddenStates, [2.0, 2.0]);
      }
    }

    hiddenStates = conv.forward(hiddenStates);
    return hiddenStates;
  }
}

class Downsample2D {
  final Normalization? norm;
  final SimpleModule conv;

  Downsample2D({this.norm, required this.conv});

  Tensor forward(Tensor hiddenStates) {
    if (norm != null) {
      hiddenStates = norm!.forward(hiddenStates.permute([0, 2, 3, 1])).permute([
        0,
        3,
        1,
        2,
      ]);
    }

    // TODO use_conv

    hiddenStates = conv.forward(hiddenStates);
    return hiddenStates;
  }

  static Future<Downsample2D> loadFromSafeTensor(
    SafeTensorLoader loader, {
    String prefix = '',
  }) async {
    // TODO
    throw UnimplementedError();
  }
}
