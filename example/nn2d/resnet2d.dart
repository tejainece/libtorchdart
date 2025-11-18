import 'package:libtorchdart/libtorchdart.dart';

void main() {
  Device device = Device(deviceType: DeviceType.cpu, deviceIndex: -1);

  final generator = Generator.getDefault(device: device);
  generator.currentSeed = 0;

  final sample = Tensor.randn([1, 32, 64, 64], device: device);
  print(sample);

  final temb = Tensor.randn([1, 128], device: device);
  print(temb);

  final resnet = ResnetBlock2D.make(numInChannels: 32, numOutChannels: 128);
  final out = resnet.forward(sample, embeds: temb);
  print(out.shape);
}
