import 'package:test/test.dart';
import 'package:libtorchdart/libtorchdart.dart';
import 'package:libtorchdart/src/autoencoder/autoencoder.dart';

void main() {
  final context = Context.best();
  group('VAE Blocks', () {
    test('DownEncoderBlock2D forward pass', () {
      final resnet = ResnetBlock2D.make(
        numInChannels: 32,
        numOutChannels: 32,
        numGroups: 32,
      );
      final downSampler = DownSample2D.make(
        numChannels: 32,
        useConv: true,
        padding: const SymmetricPadding2D.same(1),
      );

      final block = DownEncoderBlock2D(
        resnets: [resnet],
        downSamplers: [downSampler],
      );

      final input = Tensor.randn([1, 32, 64, 64]);
      final output = block.forward(input, context: context);

      // Resnet keeps shape, Downsample halves it (64 -> 32)
      expect(output.shape, [1, 32, 32, 32]);
    });

    test('VaeDecoderBlock2D forward pass', () {
      final resnet = ResnetBlock2D.make(
        numInChannels: 32,
        numOutChannels: 32,
        numGroups: 32,
      );
      final upSampler = Upsample2D.make(numChannels: 32, useConv: true);

      final block = UpDecoderBlock2D(
        resnets: [resnet],
        upsamplers: [upSampler],
      );

      final input = Tensor.randn([1, 32, 32, 32]);
      final output = block.forward(input, context: context);

      // Resnet keeps shape, Upsample doubles it (32 -> 64)
      expect(output.shape, [1, 32, 64, 64]);
    });
  });
}
