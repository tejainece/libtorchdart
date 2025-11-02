import 'package:libtorchdart/libtorchdart.dart';

enum InterpolateMode {
  nearest,
  nearestExact,
  area,
  linear,
  bilinear,
  trilinear,
  bicubic,
}

Tensor interpolateNearest(Tensor input, {int? size}) {
  // TODO

  final dim = input.dim;
  if (dim == 3) {
    // TODO
  } else if (dim == 4) {
    return upsampleNearest2D(input, outputSize: ,  scaleFactors: );
  } else if (dim == 5) {
    // TODO
  }

  throw UnimplementedError();
}
