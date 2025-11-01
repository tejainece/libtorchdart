import 'package:libtorchdart/libtorchdart.dart';
import 'package:libtorchdart/src/safetensor/storage.dart';

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
  Tensor forward(Tensor hiddenStates, {Tensor? embeds}) {
    // TODO
    throw UnimplementedError();
  }

  static Future<UNet2DMidBlock> loadFromSafeTensor(
    SafeTensorLoader loader, {
    String prefix = '',
  }) async {
    // TODO
    throw UnimplementedError();
  }
}
