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
  Tensor forward(Tensor x, {Tensor? embeds}) {
    Tensor inputTensor = x;

    Tensor hiddenStates = norm1.forward(x);
    hiddenStates = nonlinearity.forward(hiddenStates);
    print(hiddenStates);

    if (upSample != null) {
      if (hiddenStates.shape[0] >= 64) {
        inputTensor = inputTensor.contiguous();
        hiddenStates = hiddenStates.contiguous();
      }
      inputTensor = upSample!.forward(inputTensor);
      hiddenStates = upSample!.forward(hiddenStates);
    } else if (downSample != null) {
      inputTensor = downSample!.forward(inputTensor);
      hiddenStates = downSample!.forward(hiddenStates);
    }

    hiddenStates = conv1.forward(hiddenStates);

    if (embeds != null) {
      if (timeEmbProj != null) {
        if (!skipTimeAct) {
          embeds = nonlinearity.forward(embeds);
        }
        embeds = timeEmbProj!.forward(embeds).index([
          Slice.all(),
          Slice.all(),
          NewDim(),
          NewDim(),
        ]);
      }
      if (timeEmbedNorm == .def) {
        hiddenStates = hiddenStates + embeds;
        hiddenStates = norm2.forward(hiddenStates);
      } else if (timeEmbedNorm == .scaleShift) {
        final [scale, shift] = embeds.chunk(2, dim: 1);
        hiddenStates = norm2.forward(hiddenStates);
        hiddenStates = hiddenStates * (scale + 1) + shift;
      } else {
        hiddenStates = norm2.forward(hiddenStates);
      }
    } else {
      hiddenStates = norm2.forward(hiddenStates);
    }

    hiddenStates = nonlinearity.forward(hiddenStates);
    hiddenStates = dropout.forward(hiddenStates);
    hiddenStates = conv2.forward(hiddenStates);

    if (convShortcut != null) {
      inputTensor = convShortcut!.forward(inputTensor);
    }

    Tensor output = (hiddenStates + inputTensor) / outputScaleFactor;
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

  static Future<ResnetBlock2D> loadFromSafeTensor(
    SafeTensorLoader loader, {
    String prefix = '',
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
      prefix: '${prefix}norm1',
      eps: eps,
      numGroups: numGroups,
    );
    final conv1 = await Conv2D.loadFromSafeTensor(
      loader,
      prefix: '${prefix}conv1',
      padding: const SymmetricPadding2D.same(1),
    );
    final norm2 = await GroupNorm.loadFromSafeTensor(
      loader,
      prefix: '${prefix}norm2',
      eps: eps,
      numGroups: numOutGroups ?? numGroups,
    );
    final conv2 = await Conv2D.loadFromSafeTensor(
      loader,
      prefix: '${prefix}conv2',
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
          prefix: '${prefix}upsample',
          numChannels: conv1.numInChannels,
        );
      }
    } else if (down) {
      if (kernel == Resnet2DKernel.fir) {
        throw UnimplementedError();
      } else if (kernel == Resnet2DKernel.sdeVp) {
        throw UnimplementedError();
      } else {
        downSample = await Downsample2D.loadFromSafeTensor(
          loader,
          prefix: '${prefix}downsample',
          numChannels: conv1.numInChannels,
        );
      }
    }

    LinearLayer? timeEmbProj;
    /* TODO
    if (tembChannels != null) {
      if (loader.hasTensor('${prefix}time_emb_proj.weight')) {
        timeEmbProj = await LinearLayer.loadFromSafeTensor(
          loader,
          prefix: '${prefix}time_emb_proj',
        );
      }
    }
    */

    Conv2D? convShortcut;
    if (loader.hasTensor('${prefix}conv_shortcut.weight')) {
      convShortcut = await Conv2D.loadFromSafeTensor(
        loader,
        prefix: '${prefix}conv_shortcut',
      );
    }

    return ResnetBlock2D(
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
    int? tembChannels,
    bool skipTimeAct = false,
    Resnet2dTimeEmbedNormType timeEmbedNorm = Resnet2dTimeEmbedNormType.def,
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

    SimpleModule? upSample;
    SimpleModule? downSample;
    if (up) {
      if (kernel == Resnet2DKernel.fir) {
        throw UnimplementedError();
      } else if (kernel == Resnet2DKernel.sdeVp) {
        throw UnimplementedError();
      } else {
        upSample = Upsample2D.make(numChannels: numInChannels, useConv: false);
      }
    } else if (down) {
      if (kernel == Resnet2DKernel.fir) {
        throw UnimplementedError();
      } else if (kernel == Resnet2DKernel.sdeVp) {
        throw UnimplementedError();
      } else {
        downSample = Downsample2D.make(
          numChannels: numInChannels,
          useConv: false,
          padding: SymmetricPadding2D.same(1),
        );
      }
    }

    LinearLayer? timeEmbProj;
    if (tembChannels != null) {
      timeEmbProj = LinearLayer(
        weight: Tensor.empty([numOutChannels, tembChannels]),
        bias: Tensor.empty([numOutChannels]),
      );
    }

    Conv2D? convShortcut;
    if (numInChannels != numOutChannels) {
      convShortcut = Conv2D.make(
        numInChannels: numInChannels,
        numOutChannels: numOutChannels,
        kernelSize: SymmetricPadding2D.same(1),
        stride: const SymmetricPadding2D.same(1),
        padding: const SymmetricPadding2D.same(0),
      );
    }

    return ResnetBlock2D(
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

enum Resnet2dTimeEmbedNormType { none, def, scaleShift }

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
