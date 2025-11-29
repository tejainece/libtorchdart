import 'package:libtorchdart/libtorchdart.dart';

class Context {
  bool isTraining;

  Device device;

  Context({required this.isTraining, required this.device});

  factory Context.best({bool isTraining = false}) {
    return Context(isTraining: isTraining, device: Device.best());
  }
}

abstract class Module {
  String name;

  Module({required this.name});

  void resetParameters();

  Map<String, dynamic> get meta;

  Iterable<Tensor> get parameters;

  Iterable<Module> get submodules;

  @override
  String toString() {
    return '$runtimeType(${meta.entries.map((e) => '${e.key}: ${e.value}').join(', ')})';
  }
}

extension ModuleExtension on Module {
  Map<String, Tensor> stateDict({bool withName = true}) {
    String prefix = withName ? '$name.' : '';
    final ret = Map.fromEntries(
      parameters.map((e) {
        if (e.name == null) {
          throw Exception(
            'State of module $name of type $runtimeType has no name',
          );
        }
        return MapEntry(prefix + e.name!, e);
      }),
    );
    for (final submodule in submodules) {
      ret.addAll(
        submodule.stateDict().map(
          (key, value) => MapEntry(prefix + key, value),
        ),
      );
    }
    return ret;
  }
}

abstract class SimpleModule implements Module {
  Tensor forward(Tensor x, {required Context context});
}

abstract class EmbeddableModule implements Module {
  Tensor forward(Tensor x, {Tensor? embeds, required Context context});
}

abstract class InplaceModule implements Module {
  void forward_(Tensor x, {required Context context});
}
