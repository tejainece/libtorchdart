import 'package:libtorchdart/libtorchdart.dart';

class UpDecoderBlock2D extends Module implements EmbeddableModule {
  final List<ResnetBlock2D> resnets;
  final List<SimpleModule> upsamplers;

  UpDecoderBlock2D({required this.resnets, required this.upsamplers});

  @override
  Tensor forward(Tensor sample, {Tensor? embeds}) {
    for (final resnet in resnets) {
      sample = resnet.forward(sample, embeds: embeds);
    }

    for (final upsampler in upsamplers) {
      sample = upsampler.forward(sample);
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

  static Future<UpDecoderBlock2D> loadFromSafeTensor(
    SafeTensorLoader loader, {
    String prefix = '',
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

    return UpDecoderBlock2D(resnets: resnets, upsamplers: upsamplers);
  }
}
