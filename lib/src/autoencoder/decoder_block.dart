import 'package:libtorchdart/libtorchdart.dart';
import 'package:libtorchdart/src/unets/resnet2d.dart';

class VaeDecoderBlock2D {
  final List<ResnetBlock2D> resnets;
  // TODO final List<

  VaeDecoderBlock2D({required this.resnets});

  Tensor forward(Tensor sample, {Tensor? emdeds}) {
    // TODO
    throw UnimplementedError();
  }
}
