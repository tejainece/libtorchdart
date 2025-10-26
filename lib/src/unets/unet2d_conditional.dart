import 'package:libtorchdart/src/diffusion/stable_diffusion.dart';
import 'package:libtorchdart/src/safetensor/storage.dart';

/// A conditional 2D UNet. It takes a noisy sample,
/// conditional state, and a timestep and returns a sample
/// shaped output.
class UNet2DConditionModel implements UNet {
  final int inChannels;
  final int outChannels;
  final int width;
  final int height;

  final List<UNet2DDownBlock> downBlocks;
  final UNet2DMidBlock midBlock;
  final List<UNet2DUpBlock> upBlocks;

  UNet2DConditionModel({
    required this.width,
    required this.height,
    required this.inChannels,
    required this.outChannels,
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

abstract class UNet2DMidBlock implements UNet2DBlock {}
