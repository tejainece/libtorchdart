import 'package:libtorchdart/libtorchdart.dart';
import 'package:libtorchdart/src/autoencoder/autoencoder.dart';

abstract class Vae implements Module {}

/// A VAE model with KL loss for encoding images into latents and decoding latent representations into images.
class AutoencoderKL extends Module implements Vae {
  final int numInChannels;
  final int numOutChannels;
  final VaeEncoder encoder;
  final VaeDecoder decoder;

  AutoencoderKL({
    required super.name,
    required this.numInChannels,
    required this.numOutChannels,
    required this.encoder,
    required this.decoder,
  });

  // TODO

  @override
  void resetParameters() {
    encoder.resetParameters();
    decoder.resetParameters();
  }

  @override
  // TODO: implement meta
  Map<String, dynamic> get meta => throw UnimplementedError();

  @override
  late final Iterable<Tensor> parameters = [];

  @override
  Iterable<Module> get submodules => [encoder, decoder];

  static Future<AutoencoderKL> loadFromSafeTensor(
    SafeTensorLoader loader,
  ) async {
    // TODO
    throw UnimplementedError();
  }
}

/// Encoder layer of the variation autoencoder that encoder the given image (x)
/// into latent representation (z).
class VaeEncoder extends Module {
  final Conv2D convIn;
  final List<DownEncoderBlock2D> downBlocks;
  final UNet2DMidBlock midBlock;
  final GroupNorm convNormOut;
  final SiLU convActivation;
  final Conv2D convOut;

  VaeEncoder({
    super.name = 'encoder',
    required this.convIn,
    required this.downBlocks,
    required this.midBlock,
    required this.convNormOut,
    required this.convActivation,
    required this.convOut,
  });

  int get numInChannels => convIn.numInChannels;

  // TODO int get numOutChannels => convOut.numOutChannels;

  Tensor forward(
    Tensor sample, {
    Tensor? latentEmbeds,
    required Context context,
  }) {
    sample = convIn.forward(sample, context: context);

    // TODO implement grad and checkpointing
    for (int i = 0; i < downBlocks.length; i++) {
      sample = downBlocks[i].forward(sample, context: context);
    }
    sample = midBlock.forward(sample, context: context);

    sample = convNormOut.forward(sample, context: context);
    sample = convActivation.forward(sample, context: context);
    sample = convOut.forward(sample, context: context);

    return sample;
  }

  @override
  void resetParameters() {
    convIn.resetParameters();
    for (final block in downBlocks) {
      block.resetParameters();
    }
    midBlock.resetParameters();
    convNormOut.resetParameters();
    convOut.resetParameters();
  }

  @override
  late final Map<String, dynamic> meta = {
    // TODO
  };

  @override
  final Iterable<Tensor> parameters = const [];

  @override
  Iterable<Module> get submodules => [
    convIn,
    ...downBlocks,
    midBlock,
    convNormOut,
    convOut,
  ];

  static Future<VaeEncoder> loadFromSafeTensor(
    SafeTensorLoader loader, {
    String prefix = '',
    int normNumGroups = 32,
    String name = 'encoder',
    String convInName = 'conv_in',
    String midBlockName = 'mid',
    String normOutName = 'norm_out',
    String convOutName = 'conv_out',
  }) async {
    final convIn = await Conv2D.loadFromSafeTensor(
      loader,
      prefix: '$prefix$convInName',
      padding: const SymmetricPadding2D.same(1),
      stride: const SymmetricPadding2D.same(1),
    );

    final downBlocks = <DownEncoderBlock2D>[];
    // TODO downBlocks

    final midBlock = await UNet2DMidBlock.loadFromSafeTensor(
      loader,
      prefix: '$prefix$midBlockName',
      name: midBlockName,
    );

    final convNormOut = await GroupNorm.loadFromSafeTensor(
      loader,
      prefix: '$prefix$normOutName',
      numGroups: normNumGroups,
      eps: 1e-6,
      name: normOutName,
    );
    final convActivation = SiLU();
    final convOut = await Conv2D.loadFromSafeTensor(
      loader,
      prefix: '$prefix$convOutName',
      name: convOutName,
    );

    return VaeEncoder(
      name: name,
      convIn: convIn,
      downBlocks: downBlocks,
      midBlock: midBlock,
      convNormOut: convNormOut,
      convActivation: convActivation,
      convOut: convOut,
    );
  }

  static Future<VaeEncoder> make({
    required int numInChannels,
    required int numOutChannels,
    String name = 'encoder',
  }) async {
    // TODO final convIn = Conv2D.make();
    // TODO
    throw UnimplementedError();
  }
}

class VaeDecoder extends Module {
  final Conv2D convIn;
  final UNet2DMidBlock midBlock;
  final List<UpDecoderBlock2D> upBlocks;
  final Normalization convNormOut;
  final SiLU convActivation;
  final Conv2D convOut;

  VaeDecoder({
    super.name = 'decoder',
    required this.convIn,
    required this.midBlock,
    required this.upBlocks,
    required this.convNormOut,
    required this.convActivation,
    required this.convOut,
  });

  int get numInChannels => convIn.numInChannels;

  int get numOutChannels => convOut.numOutChannels;

  Tensor forward(
    Tensor sample, {
    Tensor? latentEmbeds,
    required Context context,
  }) {
    sample = convIn.forward(sample, context: context);

    // TODO implement grad and checkpointing
    sample = midBlock.forward(sample, embeds: latentEmbeds, context: context);
    for (int i = 0; i < upBlocks.length; i++) {
      sample = upBlocks[i].forward(
        sample,
        embeds: latentEmbeds,
        context: context,
      );
    }

    if (convNormOut is EmbeddableNormalizer) {
      sample = (convNormOut as EmbeddableNormalizer).forward(
        sample,
        embeds: latentEmbeds,
        context: context,
      );
    } else {
      sample = convNormOut.forward(sample, context: context);
    }

    sample = convActivation.forward(sample, context: context);
    sample = convOut.forward(sample, context: context);

    return sample;
  }

  @override
  void resetParameters() {
    convIn.resetParameters();
    midBlock.resetParameters();
    for (final block in upBlocks) {
      block.resetParameters();
    }
    convNormOut.resetParameters();
    convOut.resetParameters();
  }

  @override
  late final Map<String, dynamic> meta = {
    // TODO
  };

  @override
  late final Iterable<Tensor> parameters = [];

  @override
  late final Iterable<Module> submodules = [
    convIn,
    midBlock,
    ...upBlocks,
    convNormOut,
    convOut,
  ];

  static Future<VaeDecoder> loadFromSafeTensor(
    SafeTensorLoader loader, {
    String prefix = '',

    /// Group or spatial out normalization
    bool spatial = false,
    int normNumGroups = 32,
    String name = 'decoder',
    String convInName = 'conv_in',
    String midBlockName = 'mid',
    String normOutName = 'norm_out',
    String convOutName = 'conv_out',
  }) async {
    final convIn = await Conv2D.loadFromSafeTensor(
      loader,
      prefix: '$prefix$convInName.',
    );
    final midBlock = await UNet2DMidBlock.loadFromSafeTensor(
      loader,
      prefix: '$prefix$midBlockName.',
      name: midBlockName,
    );

    final upBlocks = <UpDecoderBlock2D>[];
    // TODO upBlocks

    Normalization convNormOut;
    if (spatial) {
      throw UnimplementedError();
    } else {
      convNormOut = await GroupNorm.loadFromSafeTensor(
        loader,
        prefix: '$prefix$normOutName.',
        name: normOutName,
        numGroups: normNumGroups,
        eps: 1e-6,
      );
    }
    final convActivation = SiLU();
    final convOut = await Conv2D.loadFromSafeTensor(
      loader,
      prefix: '$prefix$convOutName.',
      name: convOutName,
    );

    return VaeDecoder(
      name: name,
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
    String name = 'decoder',
  }) async {
    // TODO final convIn = Conv2D.make();
    // TODO
    throw UnimplementedError();
  }
}
