import 'package:libtorchdart/libtorchdart.dart';

void main() {
  final context = Context.best();

  final generator = Generator.getDefault(device: context.device);
  generator.currentSeed = 0;

  final sample = Tensor.randn([1, 32, 64, 64], device: context.device);
  final temb = Tensor.randn([1, 128], device: context.device);

  // Basic block
  final resnet = ResnetBlock2D.make(
    numInChannels: 32,
    numOutChannels: 32,
    numTembChannels: 128,
  );
  final out = resnet.forward(sample, embeds: temb, context: context);
  print(out);
}
