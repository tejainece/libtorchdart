import 'dart:math';

import 'package:tensor/tensor.dart';

class LinearLayer extends Module implements SimpleModule {
  final Tensor weight;
  final Tensor? bias;

  LinearLayer({super.name = 'linear', required this.weight, this.bias});

  int get inFeatures => weight.shape[1];

  int get outFeatures => weight.shape[0];

  @override
  Tensor forward(Tensor x, {required Context context}) {
    context.onloadModule(this);
    // Ensure input is on the same device as the weights
    final inputs = x.to(device: context.device); // TODO remove if possible
    return NNUtil.linear(inputs, weight, bias: bias);
  }

  @override
  void resetParameters() {
    Init.kaimingUniform_(weight, a: sqrt(5));
    if (bias != null) {
      final fan = Init.calculateKaimingFan(weight);
      double bound = fan.fanIn > 0 ? 1 / sqrt(fan.fanIn) : 0;
      bias!.uniform_(from: -bound, to: bound);
    }
  }

  @override
  Map<String, dynamic> get meta => {
    "inFeatures": inFeatures,
    "outFeatures": outFeatures,
    "hasBias": bias != null,
  };

  @override
  late final Iterable<Tensor> parameters = {weight, if (bias != null) bias!};

  @override
  final Iterable<Module> submodules = const [];

  Future<void> copyFromSafeTensor(
    SafeTensorLoader loader, {
    String prefix = '',
  }) async {
    if (loader.hasTensor('${prefix}weight')) {
      final newWeight = await loader.loadByName('${prefix}weight');
      weight.copy_(newWeight);
    }
    if (bias != null && loader.hasTensor('${prefix}bias')) {
      final newBias = await loader.loadByName('${prefix}bias');
      bias!.copy_(newBias);
    }
  }

  static Future<LinearLayer> loadFromSafeTensor(
    SafeTensorLoader loader, {
    String prefix = '',
    String name = 'linear',
  }) async {
    final weight = await loader.loadByName('${prefix}weight');
    Tensor? bias;
    if (loader.hasTensor('${prefix}bias')) {
      bias = await loader.loadByName('${prefix}bias');
    }
    return LinearLayer(name: name, weight: weight, bias: bias);
  }

  static LinearLayer make({
    String name = 'linear',
    required int inFeatures,
    required int outFeatures,
    bool hasBias = true,
  }) {
    final weight = Tensor.empty([outFeatures, inFeatures]);
    final bias = hasBias ? Tensor.empty([outFeatures]) : null;
    return LinearLayer(name: name, weight: weight, bias: bias)
      ..resetParameters();
  }
}
