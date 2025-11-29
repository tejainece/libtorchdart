import 'package:libtorchdart/libtorchdart.dart';

class UpDecoderBlock2D extends Module implements EmbeddableModule {
  final List<ResnetBlock2D> resnets;
  final List<SimpleModule> upsamplers;

  UpDecoderBlock2D({
    super.name = 'up_decoder_block',
    required this.resnets,
    required this.upsamplers,
  });

  @override
  Tensor forward(Tensor sample, {Tensor? embeds, required Context context}) {
    for (final resnet in resnets) {
      sample = resnet.forward(sample, embeds: embeds, context: context);
    }

    for (final upsampler in upsamplers) {
      sample = upsampler.forward(sample, context: context);
    }

    return sample;
  }

  @override
  void resetParameters() {
    for (final resnet in resnets) {
      resnet.resetParameters();
    }
    for (final upsampler in upsamplers) {
      upsampler.resetParameters();
    }
  }

  @override
  late final Map<String, dynamic> meta = {
    "resnets": resnets.map((e) => e.meta).toList(),
    "upsamplers": upsamplers.map((e) => e.meta).toList(),
  };

  @override
  final Iterable<Tensor> parameters = const [];

  @override
  late final Iterable<Module> submodules = [...resnets, ...upsamplers];

  static Future<UpDecoderBlock2D> loadFromSafeTensor(
    SafeTensorLoader loader, {
    String prefix = '',
    String name = 'up_decoder_block',
    required int numInChannels,
    required int numOutChannels,
    required int numLayers,
    bool addUpsample = true,
    double resnetEps = 1e-6,
    Activation resnetActFn = Activation.silu,
    int resnetGroups = 32,
    double dropout = 0.0,
  }) async {
    final resnets = <ResnetBlock2D>[];
    for (var i = 0; i < numLayers; i++) {
      resnets.add(
        await ResnetBlock2D.loadFromSafeTensor(
          loader,
          prefix: '${prefix}resnets.$i.',
          eps: resnetEps,
          activation: resnetActFn,
          numGroups: resnetGroups,
          dropout: dropout,
        ),
      );
    }

    final upsamplers = <SimpleModule>[];
    if (addUpsample) {
      upsamplers.add(
        await Upsample2D.loadFromSafeTensor(
          loader,
          prefix: '${prefix}upsamplers.0.',
          numChannels: numInChannels,
        ),
      );
    }

    return UpDecoderBlock2D(
      name: name,
      resnets: resnets,
      upsamplers: upsamplers,
    );
  }
}
