import 'package:libtorchdart/libtorchdart.dart';
import 'package:libtorchdart/src/unet/transformer_2d.dart';

class CrossAttnUpBlock2D extends Module implements UNet2DUpBlock {
  final List<ResnetBlock2D> resnets;
  final List<Transformer2DModel> attentions;
  final List<Upsample2D> upsamplers;

  CrossAttnUpBlock2D({
    required this.resnets,
    required this.attentions,
    required this.upsamplers,
  });

  @override
  Tensor forward(
    Tensor hiddenStates, {
    required List<Tensor> resHiddenStates,
    required Tensor timeEmbedding,
    Tensor? encoderHiddenStates,
    Tensor? attentionMask,
    Tensor? encoderAttentionMask,
    Tensor? additionalResiduals,
  }) {
    for (int i = 0; i < resnets.length; i++) {
      final resHiddenState = resHiddenStates[resHiddenStates.length - 1 - i];
      hiddenStates = Tensor.cat([hiddenStates, resHiddenState], dim: 1);

      hiddenStates = resnets[i].forward(hiddenStates, embeds: timeEmbedding);
      hiddenStates = attentions[i].forward(
        hiddenStates,
        embeds: encoderHiddenStates,
      );
    }

    for (int i = 0; i < upsamplers.length; i++) {
      hiddenStates = upsamplers[i].forward(hiddenStates);
    }

    return hiddenStates;
  }

  @override
  void resetParameters() {
    for (final resnet in resnets) {
      resnet.resetParameters();
    }
    for (final attention in attentions) {
      attention.resetParameters();
    }
    for (final upsampler in upsamplers) {
      upsampler.resetParameters();
    }
  }

  @override
  late final Map<String, dynamic> meta = {
    "resnets": resnets.map((e) => e.meta).toList(),
    "attentions": attentions.map((e) => e.meta).toList(),
    "upsamplers": upsamplers.map((e) => e.meta).toList(),
  };

  static Future<CrossAttnUpBlock2D> loadFromSafeTensor(
    SafeTensorLoader loader, {
    String prefix = '',
    required int numLayers,
    bool addUpsample = true,
    bool resnetTimeScaleShift = false,
    required int inChannels,
    required int outChannels,
    required int prevOutputChannel,
  }) async {
    final resnets = <ResnetBlock2D>[];
    final attentions = <Transformer2DModel>[];
    final upsamplers = <Upsample2D>[];

    for (int i = 0; i < numLayers; i++) {
      final resnet = await ResnetBlock2D.loadFromSafeTensor(
        loader,
        prefix: '${prefix}resnets.$i.',
      );
      resnets.add(resnet);

      final attention = await Transformer2DModel.loadFromSafeTensor(
        loader,
        prefix: '${prefix}attentions.$i.',
      );
      attentions.add(attention);
    }

    if (addUpsample) {
      final upsampler = await Upsample2D.loadFromSafeTensor(
        loader,
        prefix: '${prefix}upsamplers.0.',
        numChannels: outChannels,
      );
      upsamplers.add(upsampler);
    }

    return CrossAttnUpBlock2D(
      resnets: resnets,
      attentions: attentions,
      upsamplers: upsamplers,
    );
  }
}

class UpBlock2D extends Module implements UNet2DUpBlock {
  final List<ResnetBlock2D> resnets;
  final List<Upsample2D> upsamplers;

  UpBlock2D({required this.resnets, required this.upsamplers});

  @override
  Tensor forward(
    Tensor hiddenStates, {
    required List<Tensor> resHiddenStates,
    required Tensor timeEmbedding,
    Tensor? encoderHiddenStates,
    Tensor? attentionMask,
    Tensor? encoderAttentionMask,
    Tensor? additionalResiduals,
  }) {
    for (int i = 0; i < resnets.length; i++) {
      final resHiddenState = resHiddenStates[resHiddenStates.length - 1 - i];
      hiddenStates = Tensor.cat([hiddenStates, resHiddenState], dim: 1);

      hiddenStates = resnets[i].forward(hiddenStates, embeds: timeEmbedding);
    }

    for (int i = 0; i < upsamplers.length; i++) {
      hiddenStates = upsamplers[i].forward(hiddenStates);
    }

    return hiddenStates;
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

  static Future<UpBlock2D> loadFromSafeTensor(
    SafeTensorLoader loader, {
    String prefix = '',
    required int numLayers,
    bool addUpsample = true,
    bool resnetTimeScaleShift = false,
    required int inChannels,
    required int outChannels,
    required int prevOutputChannel,
  }) async {
    final resnets = <ResnetBlock2D>[];
    final upsamplers = <Upsample2D>[];

    for (int i = 0; i < numLayers; i++) {
      final resnet = await ResnetBlock2D.loadFromSafeTensor(
        loader,
        prefix: '${prefix}resnets.$i.',
      );
      resnets.add(resnet);
    }

    if (addUpsample) {
      final upsampler = await Upsample2D.loadFromSafeTensor(
        loader,
        prefix: '${prefix}upsamplers.0.',
        numChannels: outChannels,
      );
      upsamplers.add(upsampler);
    }

    return UpBlock2D(resnets: resnets, upsamplers: upsamplers);
  }
}
