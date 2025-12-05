import 'package:libtorchdart/libtorchdart.dart';

/// Modality-specific MLP projection for Z-Image
///
/// Projects different modality embeddings (text, VAE tokens, semantic vision)
/// to the unified transformer hidden dimension.
class ModalityProjector extends Module {
  final int inputDim;
  final int outputDim;
  final int? intermediateDim;

  final LinearLayer fc1;
  final LinearLayer? fc2;
  final Activation? activation;

  ModalityProjector({
    required super.name,
    required this.inputDim,
    required this.outputDim,
    this.intermediateDim,
    required this.fc1,
    this.fc2,
    this.activation,
  });

  @override
  Tensor forward(Tensor x, {required Context context}) {
    context.onloadModule(this);

    // First projection
    x = fc1.forward(x, context: context);

    // If two-layer MLP
    if (fc2 != null && activation != null) {
      x = activation!.forward(x, context: context);
      x = fc2!.forward(x, context: context);
    }

    return x;
  }

  @override
  void resetParameters() {
    fc1.resetParameters();
    fc2?.resetParameters();
  }

  @override
  late final Iterable<Tensor> parameters = [
    ...fc1.parameters,
    if (fc2 != null) ...fc2!.parameters,
  ];

  @override
  late final Iterable<Module> submodules = [fc1, if (fc2 != null) fc2!];

  @override
  Map<String, dynamic> get meta => {
    'inputDim': inputDim,
    'outputDim': outputDim,
    'intermediateDim': intermediateDim,
  };

  /// Create a single-layer linear projection
  static ModalityProjector makeLinear({
    required String name,
    required int inputDim,
    required int outputDim,
  }) {
    final fc1 = LinearLayer.make(
      name: 'proj',
      inFeatures: inputDim,
      outFeatures: outputDim,
    );

    return ModalityProjector(
      name: name,
      inputDim: inputDim,
      outputDim: outputDim,
      fc1: fc1,
    );
  }

  /// Create a two-layer MLP projection
  static ModalityProjector makeMLP({
    required String name,
    required int inputDim,
    required int outputDim,
    int? intermediateDim,
    String activation = 'gelu',
  }) {
    final int hiddenDim = intermediateDim ?? (inputDim + outputDim) ~/ 2;

    final fc1 = LinearLayer.make(
      name: 'fc1',
      inFeatures: inputDim,
      outFeatures: hiddenDim,
    );

    final fc2 = LinearLayer.make(
      name: 'fc2',
      inFeatures: hiddenDim,
      outFeatures: outputDim,
    );

    Activation activationFn;
    switch (activation.toLowerCase()) {
      case 'gelu':
        activationFn = GeluActivation();
        break;
      case 'silu':
      case 'swish':
        activationFn = SiLU();
        break;
      case 'relu':
        activationFn = ReLU();
        break;
      default:
        throw ArgumentError('Unsupported activation: $activation');
    }

    return ModalityProjector(
      name: name,
      inputDim: inputDim,
      outputDim: outputDim,
      intermediateDim: hiddenDim,
      fc1: fc1,
      fc2: fc2,
      activation: activationFn,
    );
  }

  /// Load from SafeTensor
  static Future<ModalityProjector> loadFromSafeTensor(
    SafeTensorLoader loader, {
    required String prefix,
    required String name,
    required int inputDim,
    required int outputDim,
    bool isTwoLayer = false,
  }) async {
    final fc1 = await LinearLayer.loadFromSafeTensor(
      loader,
      prefix: '${prefix}fc1.',
      name: 'fc1',
    );

    LinearLayer? fc2;
    if (isTwoLayer) {
      fc2 = await LinearLayer.loadFromSafeTensor(
        loader,
        prefix: '${prefix}fc2.',
        name: 'fc2',
      );
    }

    return ModalityProjector(
      name: name,
      inputDim: inputDim,
      outputDim: outputDim,
      fc1: fc1,
      fc2: fc2,
      activation: isTwoLayer ? GeluActivation() : null,
    );
  }
}
