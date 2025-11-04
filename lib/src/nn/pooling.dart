import 'package:libtorchdart/libtorchdart.dart';

class AvgPool2D extends Module implements SimpleModule {
  final SymmetricPadding2D kernelSize;
  final SymmetricPadding2D? stride;
  final SymmetricPadding2D padding;
  final bool ceilMode;
  final bool countIncludePad;
  final int? divisorOverride;

  AvgPool2D({
    required this.kernelSize,
    this.stride,
    this.padding = const SymmetricPadding2D(vertical: 0, horizontal: 0),
    this.ceilMode = false,
    this.countIncludePad = true,
    this.divisorOverride,
  });

  @override
  Tensor forward(Tensor x) {
    return avgPool2D(
      x,
      kernelSize,
      stride: stride,
      padding: padding,
      ceilMode: ceilMode,
      countIncludePad: countIncludePad,
      divisorOverride: divisorOverride,
    );
  }

  @override
  void resetParameters() {
    // TODO
    throw UnimplementedError();
  }
}
