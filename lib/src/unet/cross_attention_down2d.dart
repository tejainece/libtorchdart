import 'package:libtorchdart/libtorchdart.dart';
import 'package:libtorchdart/src/unet/transformer_2d.dart';

class CrossAttnDownBlock2D extends Module implements UNet2DDownBlock {
  final List<ResnetBlock2D> resnets;
  final List<Transformer2DModel> attentions;
  final List<DownSample2D> downSamplers;

  CrossAttnDownBlock2D({
    super.name = 'cross_attn_down_block',
    required this.resnets,
    required this.attentions,
    required this.downSamplers,
  });

  @override
  (Tensor, List<Tensor>) forward(
    Tensor hiddenStates, {
    required Tensor timeEmbedding,
    Tensor? encoderHiddenStates,
    Tensor? attentionMask,
    Tensor? encoderAttentionMask,
    Tensor? additionalResiduals,
    required Context context,
  }) {
    final outputStates = <Tensor>[];

    for (int i = 0; i < resnets.length; i++) {
      // TODO handle gradient checkpointing
      hiddenStates = resnets[i].forward(
        hiddenStates,
        embeds: timeEmbedding,
        context: context,
      );
      hiddenStates = attentions[i].forward(
        hiddenStates,
        embeds: encoderHiddenStates,
        context: context,
      );

      if (i == resnets.length - 1 && additionalResiduals != null) {
        hiddenStates = hiddenStates + additionalResiduals;
      }

      outputStates.add(hiddenStates);
    }

    for (int i = 0; i < downSamplers.length; i++) {
      hiddenStates = downSamplers[i].forward(hiddenStates, context: context);
      outputStates.add(hiddenStates);
    }

    return (hiddenStates, outputStates);
  }

  @override
  void resetParameters() {
    for (final resnet in resnets) {
      resnet.resetParameters();
    }
    for (final attention in attentions) {
      attention.resetParameters();
    }
    for (final downSampler in downSamplers) {
      downSampler.resetParameters();
    }
  }

  @override
  late final Map<String, dynamic> meta = {
    "resnets": resnets.map((e) => e.meta).toList(),
    "attentions": attentions.map((e) => e.meta).toList(),
    "downSamplers": downSamplers.map((e) => e.meta).toList(),
  };

  @override
  final Iterable<Tensor> parameters = const [];

  @override
  late final Iterable<Module> submodules = [
    ...resnets,
    ...attentions,
    ...downSamplers,
  ];

  static Future<CrossAttnDownBlock2D> loadFromSafeTensor(
    SafeTensorLoader loader, {
    String prefix = '',
    required int numLayers,
    bool addDownsample = true,
    int? downsamplePadding,
    bool resnetTimeScaleShift = false,
    required int inChannels,
    required int outChannels,
    required String name,
  }) async {
    final resnets = <ResnetBlock2D>[];
    final attentions = <Transformer2DModel>[];
    final downSamplers = <DownSample2D>[];

    for (int i = 0; i < numLayers; i++) {
      final resnet = await ResnetBlock2D.loadFromSafeTensor(
        loader,
        prefix: '${prefix}resnets.$i.',
        name: 'resnets.$i',
      );
      resnets.add(resnet);

      final attention = await Transformer2DModel.loadFromSafeTensor(
        loader,
        prefix: '${prefix}attentions.$i.',
        name: 'attentions.$i',
      );
      attentions.add(attention);
    }

    if (addDownsample) {
      final ds = await DownSample2D.loadFromSafeTensor(
        loader,
        prefix: '${prefix}downsamplers.0.',
        name: 'downsamplers.0',
        numChannels: outChannels,
      );
      downSamplers.add(ds);
    }

    return CrossAttnDownBlock2D(
      name: name,
      resnets: resnets,
      attentions: attentions,
      downSamplers: downSamplers,
    );
  }
}

class DownBlock2D extends Module implements UNet2DDownBlock {
  final List<ResnetBlock2D> resnets;
  final List<DownSample2D> downSamplers;

  DownBlock2D({
    required super.name,
    required this.resnets,
    required this.downSamplers,
  });

  @override
  (Tensor, List<Tensor>) forward(
    Tensor input, {
    required Tensor timeEmbedding,
    required Context context,
  }) {
    final outputStates = <Tensor>[];

    for (int i = 0; i < resnets.length; i++) {
      input = resnets[i].forward(
        input,
        embeds: timeEmbedding,
        context: context,
      );
      outputStates.add(input);
    }
    for (int i = 0; i < downSamplers.length; i++) {
      input = downSamplers[i].forward(input, context: context);
      outputStates.add(input);
    }
    return (input, outputStates);
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
  final Iterable<Tensor> parameters = const [];

  @override
  late final Iterable<Module> submodules = [...resnets, ...downSamplers];

  static Future<DownBlock2D> loadFromSafeTensor(
    SafeTensorLoader loader, {
    String prefix = '',
    required int numLayers,
    bool addDownsample = true,
    int? downsamplePadding,
    bool resnetTimeScaleShift = false,
    required int inChannels,
    required int outChannels,
    required String name,
  }) async {
    final resnets = <ResnetBlock2D>[];
    final downSamplers = <DownSample2D>[];

    for (int i = 0; i < numLayers; i++) {
      final resnet = await ResnetBlock2D.loadFromSafeTensor(
        loader,
        prefix: '${prefix}resnets.$i.',
        name: 'resnets.$i',
      );
      resnets.add(resnet);
    }

    if (addDownsample) {
      final ds = await DownSample2D.loadFromSafeTensor(
        loader,
        prefix: '${prefix}downsamplers.0.',
        name: 'downsamplers.0',
        numChannels: outChannels,
      );
      downSamplers.add(ds);
    }

    return DownBlock2D(
      name: name,
      resnets: resnets,
      downSamplers: downSamplers,
    );
  }
}
