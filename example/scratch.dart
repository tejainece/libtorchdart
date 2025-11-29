import 'package:libtorchdart/libtorchdart.dart';

void main() async {
  final context = Context.best();

  final file = await SafeTensorsFile.load(
    './test_data/unet/downsample/downsample_vae.safetensors',
  );
  final loader = file.mmapTensorLoader();
  final input = loader.loadByName('vae1.input');
  final output = loader.loadByName('vae1.output');

  final downsampler = await DownSample2D.loadFromSafeTensor(
    loader,
    prefix: 'vae1.downsample.',
    numChannels: input.shape[1], // Assuming NCHW layout
    padding: SymmetricPadding2D.same(0),
  );
  print(downsampler);

  final resp = downsampler.forward(input, context: context);
  print(resp[0][0][0]);
  print(output[0][0][0]);
}
