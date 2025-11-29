import 'dart:math';

import 'package:libtorchdart/libtorchdart.dart';

abstract class UNet {}

/// A conditional 2D UNet. It takes a noisy sample,
/// conditional state, and a timestep and returns a sample
/// shaped output.
class UNet2DConditionModel implements UNet {
  final int numInChannels;
  final int numOutChannels;
  final int width;
  final int height;

  final List<UNet2DDownBlock> downBlocks;
  final UNet2DMidBlock midBlock;
  final List<UNet2DUpBlock> upBlocks;

  UNet2DConditionModel({
    required this.width,
    required this.height,
    required this.numInChannels,
    required this.numOutChannels,
    required this.downBlocks,
    required this.midBlock,
    required this.upBlocks,
  });

  static Future<UNet2DConditionModel> loadFromSafeTensor(
    SafeTensorLoader loader,
  ) async {
    // TODO
    throw UnimplementedError();
  }
}

abstract class UNet2DBlock implements Module {}

abstract class UNet2DDownBlock implements UNet2DBlock {
  (Tensor, List<Tensor>) forward(
    Tensor hiddenStates, {
    required Tensor timeEmbedding,
    required Context context,
  });
}

abstract class UNet2DUpBlock implements UNet2DBlock {
  Tensor forward(
    Tensor hiddenStates, {
    required List<Tensor> resHiddenStates,
    required Tensor timeEmbedding,
    Tensor? encoderHiddenStates,
    Tensor? attentionMask,
    Tensor? encoderAttentionMask,
    Tensor? additionalResiduals,
    required Context context,
  });
}

class Resnet2DWithAttention {
  final ResnetBlock2D resnet;
  final EmbeddableModule? attention;

  Resnet2DWithAttention({required this.resnet, required this.attention});
}

class Timesteps extends Module {
  final int numChannels;
  final bool flipSinCos;
  final double downscaleFreq;

  Timesteps(
    this.numChannels, {
    super.name = '',
    this.flipSinCos = true,
    this.downscaleFreq = 0,
  });

  Tensor forward(int timestep) {
    int halfDim = numChannels ~/ 2;
    double embFactor = log(10000) / (halfDim - 1);
    var emb = Tensor.arange(halfDim, datatype: DataType.float) * -embFactor;
    emb = emb.exp();

    var t = Tensor.from([timestep.toDouble()], [1], datatype: DataType.float);
    emb = t.view([1, 1]) * emb.view([1, -1]);
    emb = Tensor.cat([emb.sin(), emb.cos()], dim: -1);

    if (numChannels % 2 == 1) {
      emb = emb.pad([0, 1]);
    }
    return emb;
  }

  @override
  void resetParameters() {}

  @override
  Map<String, dynamic> get meta => {
    "numChannels": numChannels,
    "flipSinCos": flipSinCos,
    "downscaleFreq": downscaleFreq,
  };

  @override
  final Iterable<Tensor> parameters = const [];

  @override
  final Iterable<Module> submodules = const [];
}

class TimestepEmbedding extends Module implements SimpleModule {
  final LinearLayer linear1;
  final LinearLayer linear2;
  final Activation act;

  TimestepEmbedding({
    super.name = '',
    required this.linear1,
    required this.linear2,
    required this.act,
  });

  Tensor forward(Tensor sample, {required Context context}) {
    sample = linear1.forward(sample, context: context);
    sample = act.forward(sample, context: context);
    sample = linear2.forward(sample, context: context);
    return sample;
  }

  @override
  void resetParameters() {
    linear1.resetParameters();
    linear2.resetParameters();
  }

  @override
  Map<String, dynamic> get meta => {
    "linear1": linear1.meta,
    "linear2": linear2.meta,
    "act": act.name,
  };

  @override
  final Iterable<Tensor> parameters = const [];

  @override
  late final Iterable<Module> submodules = [linear1, linear2];

  static Future<TimestepEmbedding> loadFromSafeTensor(
    SafeTensorLoader loader, {
    String prefix = '',
    required String name,
  }) async {
    final linear1 = await LinearLayer.loadFromSafeTensor(
      loader,
      prefix: '${prefix}linear_1.',
      name: 'linear_1',
    );
    final linear2 = await LinearLayer.loadFromSafeTensor(
      loader,
      prefix: '${prefix}linear_2.',
      name: 'linear_2',
    );
    // Assuming SiLU for now, but should probably be configurable or inferred
    return TimestepEmbedding(
      name: name,
      linear1: linear1,
      linear2: linear2,
      act: Activation.silu,
    );
  }
}
