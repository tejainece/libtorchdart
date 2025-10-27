import 'package:libtorchdart/src/nn/embedding_layer.dart';
import 'package:libtorchdart/src/safetensor/storage.dart';
import 'package:libtorchdart/src/tensor/tensor.dart';
import 'package:libtorchdart/src/unets/resnet2d.dart';
import 'package:libtorchdart/src/unets/unet2d_conditional.dart';

class CrossAttnDownBlock2D implements UNet2DDownBlock {
  final List<ResnetBlock2D> resnets;
  final List activation;
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
      hiddenStates = resnets[i].forward(hiddenStates, timeEmbedding);
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
}

class DownBlock2D implements UNet2DDownBlock {
  final List<ResnetBlock2D> resnets;
  final List<Downsample2D> downSamplers;

  DownBlock2D({required this.resnets, required this.downSamplers});

  Tensor forward(Tensor input, Tensor timeEmbedding) {
    for (int i = 0; i < resnets.length; i++) {
      input = resnets[i].forward(input, timeEmbedding);
    }
    for (int i = 0; i < downSamplers.length; i++) {
      input = downSamplers[i].forward(input);
    }
    return input;
  }
}

class Downsample2D {
  final SimpleModule? norm;
  final SimpleModule conv;

  Downsample2D({this.norm, required this.conv});

  Tensor forward(Tensor hiddenStates) {
    if (norm != null) {
      hiddenStates = norm!.forward(hiddenStates.permute([0, 2, 3, 1])).permute([
        0,
        3,
        1,
        2,
      ]);
    }

    // TODO use_conv

    hiddenStates = conv.forward(hiddenStates);
    return hiddenStates;
  }

  static Future<Downsample2D> loadFromSafeTensor(
    SafeTensorLoader loader, {
    String prefix = '',
  }) async {
    // TODO
    throw UnimplementedError();
  }
}
