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
    String name = 'down_block',
    String resnetPrefix = 'resnets.',
    String downsamplePrefix = 'downsamplers.',
    double resnetEps = 1e-6,
    Activation resnetActFn = Activation.silu,
    int resnetGroups = 32,
    SymmetricPadding2D downsamplePadding = const SymmetricPadding2D.same(1),
    double dropout = 0.0,
  }) async {
    final resnets = <ResnetBlock2D>[];
    int resnetIndex = 0;
    while (loader.hasTensorWithPrefix('$prefix$resnetPrefix$resnetIndex.') ||
        resnetIndex > 0) {
      resnets.add(
        await ResnetBlock2D.loadFromSafeTensor(
          loader,
          prefix: '$prefix$resnetPrefix$resnetIndex.',
          eps: resnetEps,
          activation: resnetActFn,
          numGroups: resnetGroups,
          dropout: dropout,
        ),
      );
      resnetIndex++;
    }

    if (resnets.isEmpty) {
      throw Exception('No resnets loaded');
    }

    final numOutChannels = resnets.last.conv2.numOutChannels;

    final downSamplers = <SimpleModule>[];
    int downSamplerIndex = 0;
    while (loader.hasTensorWithPrefix(
          '$prefix$downsamplePrefix$downSamplerIndex.',
        ) ||
        downSamplerIndex > 0) {
      downSamplers.add(
        await DownSample2D.loadFromSafeTensor(
          loader,
          prefix: '$prefix$downsamplePrefix$downSamplerIndex.',
          numChannels: numOutChannels,
          padding: downsamplePadding,
        ),
      );
      downSamplerIndex++;
    }

    return DownEncoderBlock2D(
      name: name,
      resnets: resnets,
      downSamplers: downSamplers,
    );
  }

  static DownEncoderBlock2D make({
    required int numInChannels,
    required int numOutChannels,
    required int numLayers,
    bool addDownsample = true,
    double resnetEps = 1e-6,
    Activation resnetActFn = Activation.silu,
    int resnetGroups = 32,
    SymmetricPadding2D downsamplePadding = const SymmetricPadding2D.same(1),
    double dropout = 0.0,
  }) {
    // TODO
    throw UnimplementedError();
  }
}
