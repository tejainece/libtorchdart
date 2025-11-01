import 'package:libtorchdart/libtorchdart.dart';

abstract class Activation {
  String get name;

  const Activation();

  Tensor forward(Tensor x);

  static const ReLU relu = ReLU();
  static const QuickGeluActivation quickGelu = QuickGeluActivation();
  static const GeluActivation gelu = GeluActivation();
  static const SiLU silu = SiLU();

  static const List<Activation> list = [quickGelu, gelu, silu, relu];

  static final Map<String, Activation> _byName = Map.fromEntries(
    list.map((v) => MapEntry(v.name, v)),
  );

  static Activation? fromName(String name) => _byName[name];
}

class ReLU implements Activation {
  @override
  String get name => "ReLU";

  const ReLU();

  @override
  Tensor forward(Tensor x) {
    return x.relu();
  }
}

class QuickGeluActivation implements Activation {
  @override
  String get name => "QuickGeLU";

  const QuickGeluActivation();

  @override
  Tensor forward(Tensor x) {
    return x * (x * 1.702).sigmoid();
  }
}

class GeluActivation implements Activation {
  @override
  String get name => "GeLU";

  const GeluActivation();

  @override
  Tensor forward(Tensor x) {
    return x.gelu(GeluApporimate.none);
  }
}

/// Applies the Sigmoid Linear Unit (SiLU) function, element-wise.
///
/// References:
///   https://arxiv.org/abs/1606.08415
///   https://arxiv.org/abs/1702.03118
///   https://arxiv.org/abs/1710.05941v1
class SiLU implements Activation {
  @override
  String get name => "SiLU";

  const SiLU();

  @override
  Tensor forward(Tensor x) {
    return x.silu();
  }
}
