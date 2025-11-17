import 'package:libtorchdart/libtorchdart.dart';

class VaeDecoderBlock2D extends Module {
  final List<ResnetBlock2D> resnets;
  // TODO final List<

  VaeDecoderBlock2D({required this.resnets});

  Tensor forward(Tensor sample, {Tensor? emdeds}) {
    // TODO
    throw UnimplementedError();
  }

  void resetParameters() {
    // TODO
    throw UnimplementedError();
  }

  @override
  late final Map<String, dynamic> meta = {
    // TODO
  };
}
