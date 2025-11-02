import 'package:libtorchdart/libtorchdart.dart';

abstract class VaeEncoderBlock2D {
  Tensor forward(Tensor sample, {Tensor? emdeds});
}
