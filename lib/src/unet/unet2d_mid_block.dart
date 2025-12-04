import 'package:libtorchdart/libtorchdart.dart';

class UNet2DMidBlock extends Module implements UNet2DBlock {
  final ResnetBlock2D resnet0;
  final List<Resnet2DWithAttention> resnets;

  UNet2DMidBlock(
    this.resnet0, {
    super.name = 'mid_block',
    required this.resnets,
  });

  Tensor forward(
    Tensor hiddenStates, {
    Tensor? embeds,
    required Context context,
  }) {
    hiddenStates = resnet0.forward(
      hiddenStates,
      embeds: embeds,
      context: context,
    );

    for (final block in resnets) {
      // TODO handle gradient checkpointing
      if (block.attention != null) {
        hiddenStates = block.attention!.forward(
          hiddenStates,
          embeds: embeds,
          context: context,
        );
      }
      hiddenStates = block.resnet.forward(
        hiddenStates,
        embeds: embeds,
        context: context,
      );
    }

    return hiddenStates;
  }

  @override
  void resetParameters() {
    resnet0.resetParameters();
    for (final block in resnets) {
      block.resnet.resetParameters();
      block.attention?.resetParameters();
    }
  }

  @override
  late final Map<String, dynamic> meta = {
    // TODO
  };

  @override
  final Iterable<Tensor> parameters = const [];

  @override
  late final Iterable<Module> submodules = [
    resnet0,
    ...resnets.map((e) => e.resnet),
    ...resnets.map((e) => e.attention).nonNulls,
  ];

  static Future<UNet2DMidBlock> loadFromSafeTensor(
    SafeTensorLoader loader, {
    String prefix = '',
    String resnetPrefix = '',
    bool resnetTimeScaleShift = false,
    bool useAttention = true,
    String name = 'mid_block',
  }) async {
    ResnetBlock2D resnet;
    final resnets = <Resnet2DWithAttention>[];

    int resnetIndex = 0;
    if (loader.hasTensor('$prefix${resnetPrefix}0.')) {
      resnetIndex = 0;
    } else if (loader.hasTensor('$prefix${resnetPrefix}1.')) {
      resnetIndex = 1;
    } else {
      throw Exception('resnet block not found');
    }

    if (resnetTimeScaleShift) {
      throw UnimplementedError();
    } else {
      resnet = await ResnetBlock2D.loadFromSafeTensor(
        loader,
        prefix: '$prefix$resnetPrefix$resnetIndex.',
        name: '$resnetPrefix$resnetIndex',
      );
    }
    resnetIndex++;

    while (true) {
      ResnetBlock2D resnetPart;
      if (resnetTimeScaleShift) {
        throw UnimplementedError();
      } else {
        if (!loader.hasTensor('$prefix$resnetPrefix$resnetIndex.')) break;
        resnetPart = await ResnetBlock2D.loadFromSafeTensor(
          loader,
          prefix: '$prefix$resnetPrefix$resnetIndex.',
          name: '$resnetPrefix$resnetIndex',
        );
      }
      EmbeddableModule? attention;
      if (useAttention) {
        // TODO
        throw UnimplementedError();
      }
      resnets.add(
        Resnet2DWithAttention(resnet: resnetPart, attention: attention),
      );
      resnetIndex++;
    }

    return UNet2DMidBlock(name: name, resnet, resnets: resnets);
  }

  static UNet2DMidBlock make({
    String name = 'mid_block',
    required int numChannels,
    int numLayers = 1,
    bool useAttention = false,
    int numGroups = 32,
    double eps = 1e-5,
    double dropout = 0.0,
    Activation nonlinearity = Activation.silu,
    int? numTembChannels,
    double outputScaleFactor = 1.0,
    bool skipTimeAct = false,
    Resnet2dTimeEmbedNormType timeEmbedNorm = Resnet2dTimeEmbedNormType.def,
  }) {
    // Create the first resnet block (resnet0)
    final resnet0 = ResnetBlock2D.make(
      name: 'resnets.0',
      numInChannels: numChannels,
      numOutChannels: numChannels,
      numGroups: numGroups,
      eps: eps,
      dropout: dropout,
      nonlinearity: nonlinearity,
      numTembChannels: numTembChannels,
      outputScaleFactor: outputScaleFactor,
      skipTimeAct: skipTimeAct,
      timeEmbedNorm: timeEmbedNorm,
    );

    // Create additional resnet blocks with optional attention
    final resnets = <Resnet2DWithAttention>[];
    for (int i = 0; i < numLayers; i++) {
      final resnetBlock = ResnetBlock2D.make(
        name: 'resnets.${i + 1}',
        numInChannels: numChannels,
        numOutChannels: numChannels,
        numGroups: numGroups,
        eps: eps,
        dropout: dropout,
        nonlinearity: nonlinearity,
        numTembChannels: numTembChannels,
        outputScaleFactor: outputScaleFactor,
        skipTimeAct: skipTimeAct,
        timeEmbedNorm: timeEmbedNorm,
      );

      EmbeddableModule? attention;
      if (useAttention) {
        // TODO: Implement attention module creation
        throw UnimplementedError('Attention modules not yet implemented');
      }

      resnets.add(
        Resnet2DWithAttention(resnet: resnetBlock, attention: attention),
      );
    }

    return UNet2DMidBlock(resnet0, name: name, resnets: resnets);
  }
}
