import 'package:libtorchdart/libtorchdart.dart';

/// A 2D upsampling layer with an optional convolution.
class Upsample2D extends Module {
  final Normalization? norm;
  final Conv2D conv;
  final bool useConvTransposed;
  final bool interpolate;
  // TODO

  Upsample2D({
    required this.norm,
    required this.conv,
    this.useConvTransposed = false,
    this.interpolate = true,
  });

  Tensor forward(Tensor hiddenStates) {
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
      // TODO
      throw UnimplementedError();
    }

    // TODO

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
