import 'package:libtorchdart/libtorchdart.dart';

abstract class Activation {
  String get name;

  const Activation();

  Tensor forward(Tensor x);

  static const QuickGeluActivation quickGelu = QuickGeluActivation();
  static const GeluActivation gelu = GeluActivation();

  static const List<Activation> list = [
    quickGelu,
    gelu,
  ];

  static final Map<String, Activation> _byName = Map.fromEntries(
    list.map((v) => MapEntry(v.name, v)),
  );

  static Activation? fromName(String name) => _byName[name];
}

class QuickGeluActivation implements Activation {
  @override
  String get name => "QuickGelu";

  const QuickGeluActivation();

  @override
  Tensor forward(Tensor x) {
    return x * (x * 1.702).sigmoid();
  }
}

class GeluActivation implements Activation {
  @override
  String get name => "Gelu";

  const GeluActivation();

  @override
  Tensor forward(Tensor x) {
    return x.gelu(
      // TODO "none"
    );
  }
}