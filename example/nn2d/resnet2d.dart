import 'package:libtorchdart/libtorchdart.dart';

void main() {
  Device device = Device(deviceType: DeviceType.cpu, deviceIndex: -1);

  final generator = Generator.getDefault(device: device);
  generator.currentSeed = 0;

  final sample = Tensor.randn([1, 32, 64, 64], device: device);
  final temb = Tensor.randn([1, 128], device: device);

  // Basic block
  final resnet = ResnetBlock2D.make(
    numInChannels: 32,
    numOutChannels: 32,
    numTembChannels: 128,
  );
  final out = resnet.forward(sample, embeds: temb);
  //print('Basic block output shape: ${out.shape}');
  //print(out);

  /*
  // Block with time embedding projection
  final resnetTemb = ResnetBlock2D.make(
    numInChannels: 32,
    numOutChannels: 128,
    tembChannels: 128,
  );
  final outTemb = resnetTemb.forward(sample, embeds: temb);
  print('Block with temb output shape: ${outTemb.shape}');

  // Block with channel mismatch (shortcut)
  final resnetShortcut = ResnetBlock2D.make(
    numInChannels: 32,
    numOutChannels: 64,
  );
  final outShortcut = resnetShortcut.forward(sample, embeds: temb);
  print('Block with shortcut output shape: ${outShortcut.shape}');
  */
}
