import 'package:libtorchdart/libtorchdart.dart';
import 'package:libtorchdart/src/safetensor/storage.dart';
import 'package:libtorchdart/src/unets/resnet2d.dart';

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

abstract class UNet2DBlock {}

abstract class UNet2DDownBlock implements UNet2DBlock {}

abstract class UNet2DUpBlock implements UNet2DBlock {}

class UNet2DMidBlock implements UNet2DBlock {
  final List<ResnetBlock2D> resnets;
  final List<Module> attentions;

  UNet2DMidBlock({required this.resnets, required this.attentions});

  Tensor forward(Tensor hiddenStates, {Tensor? embeds}) {
    // TODO
    throw UnimplementedError();
  }

  static Future<UNet2DMidBlock> loadFromSafeTensor(
    SafeTensorLoader loader, {
    String prefix = '',
    bool resnetTimeScaleShift = false,
  }) async {
    final resnets = <ResnetBlock2D>[];
    final attentions = <Module>[];

    if (resnetTimeScaleShift) {
      throw UnimplementedError();
    } else {
      resnets.add(
        ResnetBlock2D.loadFromSafeTensor(loader, prefix: '$prefix0.'),
      );
    }

    // TODO
    throw UnimplementedError();
    return UNet2DMidBlock(resnets: resnets, attentions: attentions);
  }
}
