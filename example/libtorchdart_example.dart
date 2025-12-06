import 'package:tensor/tensor.dart';

void main() {
  final tensor = Tensor.eye(
    7,
    device: Device(deviceType: DeviceType.mps, deviceIndex: -1),
  );
  print(tensor.dim);
  print(tensor.sizes);
  print(tensor.device);
  print(tensor);
}
