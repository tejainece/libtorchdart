import 'package:libtorchdart/src/nn/nn.dart';
import 'package:libtorchdart/src/tensor/tensor.dart';

class ResnetBlock2D {
  final int timeEmbeddingChannels;
  final int eps;
  final double outputScaleFactor;
  final Activation nonlinearity;
  final SimpleModule? up;
  final SimpleModule? down;

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
    required this.timeEmbeddingChannels,
    required this.eps,
    required this.outputScaleFactor,
    required this.norm1,
    required this.conv1,
    required this.norm2,
    required this.dropout,
    required this.conv2,
    required this.nonlinearity,
    required this.up,
    required this.down,
  });

  Tensor forward(Tensor x, Tensor temb) {
    Tensor hiddenStates = norm1.forward(x);
    hiddenStates = nonlinearity.forward(hiddenStates);

    // TODO upsampling
    // TODO downsampling

    hiddenStates = conv1.forward(hiddenStates);

    // TODO time emb projection

    // TODO time embedding norm

    hiddenStates = nonlinearity.forward(hiddenStates);
    hiddenStates = dropout.forward(hiddenStates);
    hiddenStates = conv2.forward(hiddenStates);

    // TODO conv shortcut

    Tensor output = (hiddenStates + x) / outputScaleFactor;
    return output;
  }

  int get numInChannels => conv1.numInChannels;

  int get numOutChannels => conv2.numOutChannels;

  int get numGroups => norm1.numGroups;

  int get numGroupsOut => norm2.numGroups;

  static Future<ResnetBlock2D> loadFromSafeTensor() {
    // TODO
    return ResnetBlock2D(
      timeEmbeddingChannels: timeEmbeddingChannels,
      eps: eps,
      outputScaleFactor: outputScaleFactor,
      norm1: norm1,
      conv1: conv1,
      norm2: norm2,
      dropout: dropout,
      conv2: conv2,
      nonlinearity: activation,
    );
  }

  static ResnetBlock2D make({
    required int numInChannels,
    required int numOutChannels,
    int? numConv2dOutChannels,
    int numGroups = 32,
    int? numOutGroups,
    double eps = 1e-5,
    double dropout = 0,
    Activation nonlinearity = Activation.silu,
  }) {
    Conv2D conv1 = Conv2D.make(
      numInChannels: numInChannels,
      numOutChannels: numOutChannels,
      kernelSize: SymmetricPadding2D.same(3),
      stride: const SymmetricPadding2D.same(1),
      padding: const SymmetricPadding2D.same(1),
    );
    Conv2D conv2 = Conv2D.make(
      numInChannels: numOutChannels,
      numOutChannels: numConv2dOutChannels ?? numOutChannels,
      kernelSize: SymmetricPadding2D.same(3),
      stride: const SymmetricPadding2D.same(1),
      padding: const SymmetricPadding2D.same(1),
    );
    GroupNorm norm1 = GroupNorm.make(
      numGroups: numGroups,
      numChannels: numInChannels,
      eps: eps,
    );
    GroupNorm norm2 = GroupNorm.make(
      numGroups: numOutGroups ?? numGroups,
      numChannels: numOutChannels,
      eps: eps,
    );
    Dropout dp = Dropout(dropout);

    SimpleModule? up;
    // TODO compute up
    SimpleModule? down;
    // TODO compute down

    // TODO
    return ResnetBlock2D(
      conv1: conv1,
      conv2: conv2,
      norm1: norm1,
      norm2: norm2,
      dropout: dp,
      nonlinearity: nonlinearity,
      up: up,
      down: down,
    );
  }
}
