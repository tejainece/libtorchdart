import 'package:libtorchdart/libtorchdart.dart';

class ResnetBlock2D extends Module implements EmbeddableModule {
  final double outputScaleFactor;
  final Activation nonlinearity;
  final SimpleModule? upSample;
  final SimpleModule? downSample;

  final GroupNorm norm1;
  final Conv2D conv1;
  final LinearLayer? timeEmbProj;
  final GroupNorm norm2;
  final Dropout dropout;
  final Conv2D conv2;

  final Conv2D? convShortcut;
  final bool skipTimeAct;
  final Resnet2dTimeEmbedNormType timeEmbedNorm;

  ResnetBlock2D({
    super.name = 'resnet',
    required this.outputScaleFactor,
    required this.norm1,
    required this.conv1,
    required this.norm2,
    required this.dropout,
    required this.conv2,
    required this.nonlinearity,
    required this.upSample,
    required this.downSample,
    required this.timeEmbProj,
    required this.convShortcut,
    required this.skipTimeAct,
    required this.timeEmbedNorm,
  });

  @override
  Tensor forward(Tensor x, {Tensor? embeds, required Context context}) {
    context.onloadModule(this);
    Tensor inputTensor = x;

    Tensor hiddenStates = norm1.forward(x, context: context);
    hiddenStates = nonlinearity.forward(hiddenStates, context: context);

    if (upSample != null) {
      if (hiddenStates.shape[0] >= 64) {
        inputTensor = inputTensor.contiguous();
        hiddenStates = hiddenStates.contiguous();
      }
      inputTensor = upSample!.forward(inputTensor, context: context);
      hiddenStates = upSample!.forward(hiddenStates, context: context);
    } else if (downSample != null) {
      inputTensor = downSample!.forward(inputTensor, context: context);
      hiddenStates = downSample!.forward(hiddenStates, context: context);
    }

    hiddenStates = conv1.forward(hiddenStates, context: context);

    if (embeds != null) {
      if (timeEmbProj != null) {
        if (!skipTimeAct) {
          embeds = nonlinearity.forward(embeds, context: context);
        }
        embeds = timeEmbProj!.forward(embeds, context: context);
        embeds = embeds.index([Slice.all(), Slice.all(), NewDim(), NewDim()]);
      }
      if (timeEmbedNorm == .def) {
        hiddenStates = hiddenStates + embeds;
        hiddenStates = norm2.forward(hiddenStates, context: context);
      } else if (timeEmbedNorm == .scaleShift) {
        final [scale, shift] = embeds.chunk(2, dim: 1);
        hiddenStates = norm2.forward(hiddenStates, context: context);
        hiddenStates = hiddenStates * (scale + 1) + shift;
      } else {
        hiddenStates = norm2.forward(hiddenStates, context: context);
      }
    } else {
      hiddenStates = norm2.forward(hiddenStates, context: context);
    }

    hiddenStates = nonlinearity.forward(hiddenStates, context: context);
    hiddenStates = dropout.forward(hiddenStates, context: context);
    hiddenStates = conv2.forward(hiddenStates, context: context);

    if (convShortcut != null) {
      inputTensor = convShortcut!.forward(inputTensor, context: context);
    }

    Tensor output = (hiddenStates + inputTensor);
    if (outputScaleFactor != 1.0) {
      output = output / outputScaleFactor;
    }

    return output;
  }

  int get numInChannels => conv1.numInChannels;

  int get numOutChannels => conv2.numOutChannels;

  int get numGroups => norm1.numGroups;

  int get numGroupsOut => norm2.numGroups;

  @override
  void resetParameters() {
    norm1.resetParameters();
    conv1.resetParameters();
    norm2.resetParameters();
    dropout.resetParameters();
    conv2.resetParameters();
    upSample?.resetParameters();
    downSample?.resetParameters();
    timeEmbProj?.resetParameters();
    convShortcut?.resetParameters();
  }

  @override
  late final Map<String, dynamic> meta = {
    "outputScaleFactor": outputScaleFactor,
    "norm1": norm1.meta,
    "conv1": conv1.meta,
    "norm2": norm2.meta,
    "dropout": dropout.meta,
    "conv2": conv2.meta,
    "nonlinearity": nonlinearity.toString(),
    "upSample": upSample?.meta,
    "downSample": downSample?.meta,
    "timeEmbProj": timeEmbProj?.meta,
    "convShortcut": convShortcut?.meta,
  };

  @override
  late final Iterable<Tensor> parameters = [];

  @override
  late final Iterable<Module> submodules = [
    // TODO nonlinearity,
    norm1,
    conv1,
    norm2,
    dropout,
    conv2,
    if (upSample != null) upSample!,
    if (downSample != null) downSample!,
    if (timeEmbProj != null) timeEmbProj!,
    if (convShortcut != null) convShortcut!,
  ];

  static Future<ResnetBlock2D> loadFromSafeTensor(
    SafeTensorLoader loader, {
    String prefix = '',
    String name = 'resnet',
    ResnetBlock2DSubmoduleNames submoduleNames =
        const ResnetBlock2DSubmoduleNames(),
    double eps = 1e-5,
    double outputScaleFactor = 1.0,
    Activation activation = Activation.silu,
    int numGroups = 32,
    int? numOutGroups,
    double dropout = 0,
    Resnet2DKernel kernel = Resnet2DKernel.none,
    bool up = false,
    bool down = false,
    bool skipTimeAct = false,
    Resnet2dTimeEmbedNormType timeEmbedNorm = Resnet2dTimeEmbedNormType.def,
  }) async {
    final norm1 = await GroupNorm.loadFromSafeTensor(
      loader,
      prefix: '$prefix${submoduleNames.norm1}.',
      name: submoduleNames.norm1,
      eps: eps,
      numGroups: numGroups,
    );
    final conv1 = await Conv2D.loadFromSafeTensor(
      loader,
      prefix: '$prefix${submoduleNames.conv1}.',
      name: submoduleNames.conv1,
      padding: const SymmetricPadding2D.same(1),
    );
    final norm2 = await GroupNorm.loadFromSafeTensor(
      loader,
      prefix: '$prefix${submoduleNames.norm2}.',
      name: submoduleNames.norm2,
      eps: eps,
      numGroups: numOutGroups ?? numGroups,
    );
    final conv2 = await Conv2D.loadFromSafeTensor(
      loader,
      prefix: '$prefix${submoduleNames.conv2}.',
      name: submoduleNames.conv2,
      padding: const SymmetricPadding2D.same(1),
    );
    final dropoutLayer = Dropout(dropout);

    SimpleModule? upSample;
    SimpleModule? downSample;
    if (up) {
      if (kernel == Resnet2DKernel.fir) {
        throw UnimplementedError();
      } else if (kernel == Resnet2DKernel.sdeVp) {
        throw UnimplementedError();
      } else {
        upSample = await Upsample2D.loadFromSafeTensor(
          loader,
          prefix: '${prefix}upsample.',
          numChannels: conv1.numInChannels,
        );
      }
    } else if (down) {
      if (kernel == Resnet2DKernel.fir) {
        throw UnimplementedError();
      } else if (kernel == Resnet2DKernel.sdeVp) {
        throw UnimplementedError();
      } else {
        downSample = await DownSample2D.loadFromSafeTensor(
          loader,
          prefix: '${prefix}downsample.',
          numChannels: conv1.numInChannels,
        );
      }
    }

    LinearLayer? timeEmbProj;
    if (loader.hasTensorWithPrefix('${prefix}time_emb_proj.')) {
      timeEmbProj = await LinearLayer.loadFromSafeTensor(
        loader,
        prefix: '${prefix}time_emb_proj.',
      );
    }

    Conv2D? convShortcut;
    if (loader.hasTensorWithPrefix('${prefix}conv_shortcut.')) {
      convShortcut = await Conv2D.loadFromSafeTensor(
        loader,
        prefix: '${prefix}conv_shortcut.',
        padding: const SymmetricPadding2D.same(0),
      );
    }

    return ResnetBlock2D(
      name: name,
      outputScaleFactor: outputScaleFactor,
      norm1: norm1,
      conv1: conv1,
      norm2: norm2,
      dropout: dropoutLayer,
      conv2: conv2,
      nonlinearity: activation,
      upSample: upSample,
      downSample: downSample,
      timeEmbProj: timeEmbProj,
      convShortcut: convShortcut,
      skipTimeAct: skipTimeAct,
      timeEmbedNorm: timeEmbedNorm,
    );
  }

  static ResnetBlock2D make({
    String name = 'resnet',
    ResnetBlock2DSubmoduleNames submoduleNames =
        const ResnetBlock2DSubmoduleNames(),
    required int numInChannels,
    required int numOutChannels,
    int? numConv2dOutChannels,
    int numGroups = 32,
    int? numOutGroups,
    double eps = 1e-5,
    double dropout = 0,
    Activation nonlinearity = Activation.silu,
    Resnet2DKernel kernel = Resnet2DKernel.none,
    bool up = false,
    bool down = false,
    double outputScaleFactor = 1.0,
    int? numTembChannels,
    bool skipTimeAct = false,
    Resnet2dTimeEmbedNormType timeEmbedNorm = Resnet2dTimeEmbedNormType.def,
  }) {
    GroupNorm norm1 = GroupNorm.make(
      name: submoduleNames.norm1,
      numGroups: numGroups,
      numChannels: numInChannels,
      eps: eps,
    );
    Conv2D conv1 = Conv2D.make(
      name: submoduleNames.conv1,
      numInChannels: numInChannels,
      numOutChannels: numOutChannels,
      kernelSize: SymmetricPadding2D.same(3),
      stride: const SymmetricPadding2D.same(1),
      padding: const SymmetricPadding2D.same(1),
    );

    LinearLayer? timeEmbProj;
    if (numTembChannels != null) {
      if (timeEmbedNorm == .def) {
        timeEmbProj = LinearLayer.make(
          name: submoduleNames.timeEmbProj,
          inFeatures: numTembChannels,
          outFeatures: numOutChannels,
        );
      } else if (timeEmbedNorm == .scaleShift) {
        timeEmbProj = LinearLayer.make(
          name: submoduleNames.timeEmbProj,
          inFeatures: numTembChannels,
          outFeatures: numOutChannels * 2,
        );
      } else {
        throw UnimplementedError(
          'Unknown time embedding norm type: $timeEmbedNorm',
        );
      }
    }

    GroupNorm norm2 = GroupNorm.make(
      name: submoduleNames.norm2,
      numGroups: numOutGroups ?? numGroups,
      numChannels: numOutChannels,
      eps: eps,
    );
    Dropout dp = Dropout(dropout);

    Conv2D conv2 = Conv2D.make(
      name: submoduleNames.conv2,
      numInChannels: numOutChannels,
      numOutChannels: numConv2dOutChannels ?? numOutChannels,
      kernelSize: SymmetricPadding2D.same(3),
      stride: const SymmetricPadding2D.same(1),
      padding: const SymmetricPadding2D.same(1),
    );

    SimpleModule? upSample;
    SimpleModule? downSample;
    if (up) {
      if (kernel == Resnet2DKernel.fir) {
        throw UnimplementedError();
      } else if (kernel == Resnet2DKernel.sdeVp) {
        throw UnimplementedError();
      } else {
        upSample = Upsample2D.make(
          name: submoduleNames.upSample,
          numChannels: numInChannels,
          useConv: false,
        );
      }
    } else if (down) {
      if (kernel == Resnet2DKernel.fir) {
        throw UnimplementedError();
      } else if (kernel == Resnet2DKernel.sdeVp) {
        throw UnimplementedError();
      } else {
        downSample = DownSample2D.make(
          numChannels: numInChannels,
          useConv: false,
          padding: SymmetricPadding2D.same(1),
        );
      }
    }

    Conv2D? convShortcut;
    if (numInChannels != numOutChannels) {
      convShortcut = Conv2D.make(
        name: submoduleNames.convShortcut,
        numInChannels: numInChannels,
        numOutChannels: numOutChannels,
        kernelSize: SymmetricPadding2D.same(1),
        stride: const SymmetricPadding2D.same(1),
        padding: const SymmetricPadding2D.same(0),
      );
    }

    return ResnetBlock2D(
      name: name,
      conv1: conv1,
      conv2: conv2,
      norm1: norm1,
      norm2: norm2,
      dropout: dp,
      nonlinearity: nonlinearity,
      upSample: upSample,
      downSample: downSample,
      outputScaleFactor: outputScaleFactor,
      timeEmbProj: timeEmbProj,
      convShortcut: convShortcut,
      skipTimeAct: skipTimeAct,
      timeEmbedNorm: timeEmbedNorm,
    );
  }
}

class ResnetBlock2DSubmoduleNames {
  final String norm1;
  final String conv1;
  final String norm2;
  final String dropout;
  final String conv2;
  final String upSample;
  final String downSample;
  final String timeEmbProj;
  final String convShortcut;

  const ResnetBlock2DSubmoduleNames({
    this.norm1 = 'norm1',
    this.conv1 = 'conv1',
    this.norm2 = 'norm2',
    this.dropout = 'dropout',
    this.conv2 = 'conv2',
    this.upSample = 'upSample',
    this.downSample = 'downSample',
    this.timeEmbProj = 'timeEmbProj',
    this.convShortcut = 'convShortcut',
  });
}

enum Resnet2dTimeEmbedNormType { def, scaleShift }

class Resnet2DKernel {
  final String name;

  const Resnet2DKernel(this.name);

  static const Resnet2DKernel none = Resnet2DKernel('none');
  static const Resnet2DKernel fir = Resnet2DKernel('fir');
  static const Resnet2DKernel sdeVp = Resnet2DKernel('sde_vp');

  static final List<Resnet2DKernel> values = [none, fir, sdeVp];

  static final Map<String, Resnet2DKernel> map = values.asMap().map(
    (key, value) => MapEntry(value.name, value),
  );
}
