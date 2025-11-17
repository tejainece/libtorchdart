import 'dart:math';

import 'package:libtorchdart/libtorchdart.dart';

abstract class Init {
  static void uniform_({
    required Tensor tensor,
    required double from,
    required double to,
    Generator? generator,
  }) {
    tensor.uniform_(from: from, to: to, generator: generator);
  }

  static void kaimingUniform_(
    Tensor tensor, {
    double a = 0,
    bool fanIn = true,
    KaimingNonLinearity nonLinearity = KaimingNonLinearity.relu,
    Generator? generator,
  }) {
    final (:fanIn, :fanOut) = calculateKaimingFan(tensor);
    final double gain = calculateGain(nonLinearity, a);
    double std = gain / sqrt(fanIn);
    double bound = sqrt(3.0) * std;
    // TODO no_grad
    tensor.uniform_(from: -bound, to: bound, generator: generator);
  }

  static ({int fanIn, int fanOut}) calculateKaimingFan(Tensor tensor) {
    final shape = tensor.shape;
    if (shape.length < 2) {
      throw Exception(
        "Fan in and fan out cannot be computed for tensor with fewer than 2 dimensions",
      );
    }

    int numInputFMaps = shape[1];
    int numOutputFMaps = shape[0];
    int receptiveFIeldSize = 1;
    for (int i = 2; i < shape.length; i++) {
      receptiveFIeldSize *= shape[i];
    }
    return (
      fanIn: numInputFMaps * receptiveFIeldSize,
      fanOut: numOutputFMaps * receptiveFIeldSize,
    );
  }
}

enum KaimingNonLinearity {
  linear,
  conv1d,
  conv2d,
  conv3d,
  convTranspose1d,
  convTranspose2d,
  convTranspose3d,
  sigmoid,
  tanh,
  relu,
  leakyRelu,
  selu,
}

double calculateGain(KaimingNonLinearity nonLinearity, double? param) {
  switch (nonLinearity) {
    case KaimingNonLinearity.linear ||
        KaimingNonLinearity.conv1d ||
        KaimingNonLinearity.conv2d ||
        KaimingNonLinearity.conv3d ||
        KaimingNonLinearity.convTranspose1d ||
        KaimingNonLinearity.convTranspose2d ||
        KaimingNonLinearity.convTranspose3d:
      return 1.0;
    case KaimingNonLinearity.tanh:
      return 5.0 / 3.0;
    case KaimingNonLinearity.relu:
      return sqrt(2.0);
    case KaimingNonLinearity.leakyRelu:
      double negativeSlope = param ?? 0.01;
      return sqrt(2.0 / (1 + negativeSlope * negativeSlope));
    case KaimingNonLinearity.selu:
      return 3.0 / 4.0;
    default:
      throw Exception("Unsupported non-linearity: $nonLinearity");
  }
}
