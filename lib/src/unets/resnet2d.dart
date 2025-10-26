import 'package:libtorchdart/src/nn/nn.dart';
import 'package:libtorchdart/src/tensor/tensor.dart';

class ResnetBlock2D {
  final int inChannels;
  final int outChannels;
  final int timeEmbeddingChannels;
  final int numGroups;
  final int numGroupsOut;
  final int eps;
  final double outputScaleFactor;
  final Activation activation;
  /*
  /// If true, adds an upsampling layer
  final bool up;

  /// If true, adds a downsampling layer
  final bool down;*/

  final GroupNorm norm1;
  final Conv2D conv1;
  // TODO time embedding proj
  final GroupNorm norm2;
  final Dropout dropout;
  final Conv2D conv2;

  // TODO upsample
  // TODO downsample

  // TODO conv shortcut

  ResnetBlock2D({
    required this.inChannels,
    required this.outChannels,
    required this.timeEmbeddingChannels,
    required this.numGroups,
    required this.numGroupsOut,
    required this.eps,
    required this.outputScaleFactor,
    required this.norm1,
    required this.conv1,
    required this.norm2,
    required this.dropout,
    required this.conv2,
    required this.activation,
  });

  Tensor forward(Tensor x, Tensor temb) {
    Tensor hiddenStates = norm1.forward(x);
    hiddenStates = activation.forward(hiddenStates);

    // TODO upsampling
    // TODO downsampling

    hiddenStates = conv1.forward(hiddenStates);

    // TODO time emb projection

    // TODO time embedding norm

    hiddenStates = activation.forward(hiddenStates);
    hiddenStates = dropout.forward(hiddenStates);
    hiddenStates = conv2.forward(hiddenStates);

    // TODO conv shortcut

    Tensor output = (hiddenStates + x) / outputScaleFactor;
    return output;
  }
}
