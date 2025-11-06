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

abstract class UNet2DBlock implements Module {}

abstract class UNet2DDownBlock implements UNet2DBlock {}

abstract class UNet2DUpBlock implements UNet2DBlock {}

class UNet2DMidBlock extends Module implements UNet2DBlock {
  final ResnetBlock2D resnet;
  final List<Resnet2DWithAttention> resnets;

  UNet2DMidBlock(this.resnet, {required this.resnets});

  Tensor forward(Tensor hiddenStates, {Tensor? embeds}) {
    hiddenStates = resnet.forward(hiddenStates, embeds: embeds);
    for (final block in resnets) {
      // TODO handle gradient checkpointing
      if (block.attention != null) {
        hiddenStates = block.attention!.forward(hiddenStates, embeds: embeds);
      }
      hiddenStates = block.resnet.forward(hiddenStates, embeds: embeds);
    }

    return hiddenStates;
  }

  @override
  void resetParameters() {
    resnet.resetParameters();
    for (final block in resnets) {
      block.resnet.resetParameters();
      block.attention?.resetParameters();
    }
  }

  static Future<UNet2DMidBlock> loadFromSafeTensor(
    SafeTensorLoader loader, {
    String prefix = '',
    String resnetPrefix = '',
    bool resnetTimeScaleShift = false,
    bool useAttention = true,
  }) async {
    final resnets = <ResnetBlock2D>[];
    final attentions = <Module>[];

    int i = 0;
    if (loader.hasTensor('$prefix${resnetPrefix}0.')) {
      i = 0;
    } else if (loader.hasTensor('$prefix${resnetPrefix}1.')) {
      i = 1;
    } else {
      throw Exception('resnet block not found');
    }
    if (resnetTimeScaleShift) {
      throw UnimplementedError();
    } else {
      resnets.add(
        await ResnetBlock2D.loadFromSafeTensor(
          loader,
          prefix: '$prefix$resnetPrefix$i.',
          
        ),
      );
    }

    // TODO load layers
    int layerId = 0;
    // TODO
    while (true) {
      if (useAttention) {
        // TODO
        throw UnimplementedError();
      }
      if (resnetTimeScaleShift) {
        throw UnimplementedError();
      } else {
        resnets.add(
          await ResnetBlock2D.loadFromSafeTensor(loader, prefix: '$prefix.'),
        );
      }
      layerId++;
    }

    // TODO
    throw UnimplementedError();
    return UNet2DMidBlock(resnets: resnets, attentions: attentions);
  }
}

class Resnet2DWithAttention {
  final ResnetBlock2D resnet;
  final EmbeddableModule? attention;

  Resnet2DWithAttention({required this.resnet, required this.attention});
}
