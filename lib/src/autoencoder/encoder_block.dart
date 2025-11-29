import 'package:libtorchdart/libtorchdart.dart';

class DownEncoderBlock2D extends Module implements EmbeddableModule {
  final List<ResnetBlock2D> resnets;
  final List<SimpleModule> downSamplers;

  DownEncoderBlock2D({
    super.name = 'down_block',
    required this.resnets,
    required this.downSamplers,
  });

  @override
  Tensor forward(Tensor sample, {Tensor? embeds, required Context context}) {
    for (final resnet in resnets) {
      sample = resnet.forward(sample, embeds: embeds, context: context);
    }

    for (final downSampler in downSamplers) {
      sample = downSampler.forward(sample, context: context);
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

  @override
  late final Iterable<Tensor> parameters = [];

  @override
  late final Iterable<Module> submodules = [...resnets, ...downSamplers];

  static Future<DownEncoderBlock2D> loadFromSafeTensor(
    SafeTensorLoader loader, {
    String prefix = '',
    bool addDownsample = true,
    double resnetEps = 1e-6,
    Activation resnetActFn = Activation.silu,
    int resnetGroups = 32,
    SymmetricPadding2D downsamplePadding = const SymmetricPadding2D.same(1),
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
    // TODO verify that resnets are not empty

    final numOutChannels = resnets.last.conv2.numOutChannels;

    final downSamplers = <SimpleModule>[];
    if (addDownsample) {
      downSamplers.add(
        await DownSample2D.loadFromSafeTensor(
          loader,
          prefix: '${prefix}downsamplers.0.',
          numChannels: numOutChannels,
          padding: downsamplePadding,
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
