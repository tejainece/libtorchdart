import 'dart:math';

import 'package:tensor/tensor.dart';

void main() {
  Device device = Device(deviceType: DeviceType.cpu, deviceIndex: -1);

  final generator = Generator.getDefault(device: device);
  generator.currentSeed = 0;

  final weights = Tensor.empty([32, 32, 3, 3], device: device);
  Init.kaimingUniform_(weights, a: sqrt(5), generator: generator);
  print(weights);
}
