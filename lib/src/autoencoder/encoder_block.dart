import 'package:libtorchdart/libtorchdart.dart';

class DownEncoderBlock2D extends Module implements EmbeddableModule {
  final List<ResnetBlock2D> resnets;
  final List<SimpleModule> downSamplers;

  DownEncoderBlock2D({required this.resnets, required this.downSamplers});

  @override
  Tensor forward(Tensor sample, {Tensor? embeds}) {
    for (final resnet in resnets) {
      sample = resnet.forward(sample, embeds: embeds);
    }

    for (final downSampler in downSamplers) {
      sample = downSampler.forward(sample);
    }

    return sample;
  }

  @override
  void resetParameters() {
    for (final resnet in resnets) {
      resnet.resetParameters();
    }
    for (final downSampler in downSamplers) {
      downSampler.resetParameters();
    }
  }

  @override
  late final Map<String, dynamic> meta = {
    "resnets": resnets.map((e) => e.meta).toList(),
    "downSamplers": downSamplers.map((e) => e.meta).toList(),
  };

  static Future<DownEncoderBlock2D> loadFromSafeTensor(
    SafeTensorLoader loader, {
    String prefix = '',
    bool addDownsample = true,
    double resnetEps = 1e-6,
    Activation resnetActFn = Activation.silu,
    int resnetGroups = 32,
    bool downsamplePadding = true,
    double dropout = 0.0,
  }) async {
    final resnets = <ResnetBlock2D>[];
    for (var i = 0; true; i++) {
      final name = '${prefix}resnets.$i.';
      if (!loader.hasTensorWithPrefix(name)) break;
      resnets.add(
        await ResnetBlock2D.loadFromSafeTensor(
          loader,
          prefix: name,
          eps: resnetEps,
          activation: resnetActFn,
          numGroups: resnetGroups,
          dropout: dropout,
        ),
      );
    }

    final downSamplers = <SimpleModule>[];
    if (addDownsample) {
      downSamplers.add(
        await Downsample2D.loadFromSafeTensor(
          loader,
          prefix: '${prefix}downsamplers.0.',
          numChannels: numOutChannels,
          padding: downsamplePadding
              ? const SymmetricPadding2D.same(1)
              : const SymmetricPadding2D.same(0),
        ),
      );
    }

    return DownEncoderBlock2D(resnets: resnets, downSamplers: downSamplers);
  }

  static DownEncoderBlock2D make({
    required int numInChannels,
    required int numOutChannels,
    required int numLayers,
  }) {
    // TODO
    throw UnimplementedError();
  }
}
