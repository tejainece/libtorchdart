import 'package:libtorchdart/src/nn/embedding_layer.dart';
import 'package:libtorchdart/src/tensor/tensor.dart';
import 'package:libtorchdart/src/unet/resnet2d.dart';
import 'package:libtorchdart/src/unet/sampler.dart';
import 'package:libtorchdart/src/unet/unet2d_conditional.dart';

class CrossAttnDownBlock2D extends Module implements UNet2DDownBlock {
  final List<ResnetBlock2D> resnets;
  final List<SimpleModule> activation;
  final List<Downsample2D> downSamplers;

  CrossAttnDownBlock2D({
    required this.resnets,
    required this.activation,
    required this.downSamplers,
  });

  Tensor forward(
    Tensor hiddenStates, {
    required Tensor timeEmbedding,
    Tensor? encoderHiddenStates,
    Tensor? attentionMask,
    Tensor? encoderAttentionMask,
    Tensor? additionalResiduals,
  }) {
    // TODO

    // TODO output states

    for (int i = 0; i < resnets.length; i++) {
      // TODO handle gradient checkpointing
      hiddenStates = resnets[i].forward(hiddenStates, embeds: timeEmbedding);
      // TODO activation

      if (i == resnets.length - 1 && additionalResiduals != null) {
        hiddenStates = hiddenStates + additionalResiduals;
      }
    }

    for (int i = 0; i < downSamplers.length; i++) {
      hiddenStates = downSamplers[i].forward(hiddenStates);
    }

    return hiddenStates;
  }

  @override
  void resetParameters() {
    // TODO
    throw UnimplementedError();
  }
}

class DownBlock2D extends Module implements UNet2DDownBlock {
  final List<ResnetBlock2D> resnets;
  final List<Downsample2D> downSamplers;

  DownBlock2D({required this.resnets, required this.downSamplers});

  Tensor forward(Tensor input, Tensor timeEmbedding) {
    for (int i = 0; i < resnets.length; i++) {
      input = resnets[i].forward(input, embeds: timeEmbedding);
    }
    for (int i = 0; i < downSamplers.length; i++) {
      input = downSamplers[i].forward(input);
    }
    return input;
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
}
