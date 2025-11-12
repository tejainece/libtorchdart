import 'package:libtorchdart/libtorchdart.dart';

class DownEncoderBlock2D extends Module implements EmbeddableModule {
  final List<ResnetBlock2D> resnets;
  final List<SimpleModule> downSamplers;

  DownEncoderBlock2D({required this.resnets, required this.downSamplers});

  @override
  Tensor forward(Tensor sample, {Tensor? embeds}) {
    for (final resnet in resnets) {
      sample = resnet.forward(sample, embeds: embeds);
    }

    for (final doenSampler in downSamplers) {
      sample = doenSampler.forward(sample);
    }

    return sample;
  }

  @override
  void resetParameters() {
    throw UnimplementedError();
  }

  static Future<DownEncoderBlock2D> loadFromSafeTensor(
    SafeTensorLoader loader, {
    String path = '',
  }) async {
    final resnets = <ResnetBlock2D>[];
    // TODO resnets

    final downSamplers = <SimpleModule>[];
    // TODO down samplers

    return DownEncoderBlock2D(resnets: resnets, downSamplers: downSamplers);
  }
}
