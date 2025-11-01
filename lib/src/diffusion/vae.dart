import 'package:libtorchdart/libtorchdart.dart';
import 'package:libtorchdart/src/unets/unet2d_conditional.dart';

abstract class Vae {}

/// A VAE model with KL loss for encoding images into latents and decoding latent representations into images.
class AutoencoderKL implements Vae {
  final int numInChannels;
  final int numOutChannels;
  final VaeEncoder encoder;
  final VaeDecoder decoder;

  AutoencoderKL({
    required this.numInChannels,
    required this.numOutChannels,
    required this.encoder,
    required this.decoder,
  });

  // TODO

  static Future<AutoencoderKL> loadFromSafeTensor(
    SafeTensorLoader loader,
  ) async {
    // TODO
    throw UnimplementedError();
  }
}

class VaeEncoder {}

class VaeDecoder extends Module {
  final Conv2D convIn;
  final UNet2DMidBlock midBlock;
  final List<VaeDecoderBlock2D> upBlocks;
  final Normalization convNormOut;
  final SiLU convActivation;
  final Conv2D convOut;

  VaeDecoder({
    required this.convIn,
    required this.midBlock,
    required this.upBlocks,
    required this.convNormOut,
    required this.convActivation,
    required this.convOut,
  });

  int get numInChannels => convIn.numInChannels;

  int get numOutChannels => convOut.numOutChannels;

  Tensor forward(Tensor sample, {Tensor? latentEmbeds}) {
    sample = convIn.forward(sample);

    // TODO implement grad and checkpointing
    sample = midBlock.forward(sample, embeds: latentEmbeds);
    for (int i = 0; i < upBlocks.length; i++) {
      sample = upBlocks[i].forward(sample, emdeds: latentEmbeds);
    }

    if (convNormOut is EmbeddableNormalizer) {
      sample = (convNormOut as EmbeddableNormalizer).forward(
        sample,
        embeds: latentEmbeds,
      );
    } else {
      sample = convNormOut.forward(sample);
    }

    sample = convActivation.forward(sample);
    sample = convOut.forward(sample);

    return sample;
  }

  static Future<VaeDecoder> loadFromSafeTensor(
    SafeTensorLoader loader, {
    String prefix = '',

    /// Group or spatial out normalization
    bool spatial = false,
    int normNumGroups = 32,
  }) async {
    final convIn = await Conv2D.loadFromSafeTensor(
      loader,
      prefix: '${prefix}conv_in',
    );
    final midBlock = await UNet2DMidBlock.loadFromSafeTensor(
      loader,
      prefix: '${prefix}mid',
    );

    final upBlocks = <VaeDecoderBlock2D>[];
    // TODO upBLocks

    Normalization convNormOut;
    if (spatial) {
      throw UnimplementedError();
    } else {
      convNormOut = await GroupNorm.loadFromSafeTensor(
        loader,
        prefix: '${prefix}norm_out',
        numGroups: normNumGroups,
        eps: 1e-6,
      );
    }
    final convActivation = SiLU();
    final convOut = await Conv2D.loadFromSafeTensor(
      loader,
      prefix: '${prefix}conv_out',
    );

    return VaeDecoder(
      convIn: convIn,
      midBlock: midBlock,
      upBlocks: upBlocks,
      convNormOut: convNormOut,
      convActivation: convActivation,
      convOut: convOut,
    );
  }

  static Future<VaeDecoder> make({
    required int numInChannels,
    required int numOutChannels,
  }) async {
    final convIn = Conv2D.make();
    // TODO
    throw UnimplementedError();
  }
}

abstract class VaeEncoderBlock2D {}

abstract class VaeDecoderBlock2D {
  Tensor forward(Tensor sample, {Tensor? emdeds});
}
