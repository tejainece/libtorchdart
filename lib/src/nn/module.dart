import 'package:libtorchdart/libtorchdart.dart';

abstract class Module {
  String name;

  Module({required this.name});

  void resetParameters();

  Map<String, dynamic> get meta;

  Iterable<Tensor> get parameters;

  Iterable<Module> get submodules;

  void to_(Device device, {bool cascade = false}) {
    for (final parameter in parameters) {
      if (parameter.device == device) continue;
      parameter.to_(device: device);
    }
    if (cascade) {
      for (final submodule in submodules) {
        submodule.to_(device, cascade: cascade);
      }
    }
  }

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

class Context {
  bool isTraining;

  Device device;

  final Offloader offloader = Offloader();

  Context({required this.isTraining, required this.device});

  factory Context.best({bool isTraining = false}) {
    return Context(isTraining: isTraining, device: Device.best());
  }

  void onloadModule(Module module) {
    if (device == Device.cpu) {
      // TODO handle low RAM situations
      return;
    }
    offloader.freeAndLoadModule(module, device);
  }
}

class Offloader {
  final Set<Module> keep = {};

  final Set<Module> modules = {};

  Offloader();

  void freeAndLoadModule(Module module, Device device) {
    if (modules.contains(module)) return;

    int requiredMemory = module.parameters.fold(0, (previousValue, element) {
      if (element.device == device) return previousValue;
      return previousValue + element.elementSize;
    });
    if (requiredMemory > device.freeMemory) {
      if (!freeMemory(requiredMemory, device)) {
        throw Exception('Not enough memory');
      }
    }
    module.to_(device);
    modules.add(module);
  }

  void offloadModule(Module module) {
    if (!modules.contains(module)) return;
    for (final parameter in module.parameters) {
      final device = parameter.device;
      if (device == Device.cpu) continue;
      // TODO pin tensor?
      parameter.to_(device: Device.cpu);
    }
    modules.remove(module);
    keep.remove(module);
  }

  bool freeMemory(int requiredMemory, Device device) {
    // TODO implement an intelligent algorithm to decide which modules to offload
    // TODO better to offload lowest memory modules first?
    for (final module in modules) {
      if (keep.contains(module)) continue;
      offloadModule(module);
      if (device.freeMemory >= requiredMemory) return true;
    }
    return false;
  }

  void offloadAll() {
    for (final module in modules) {
      offloadModule(module);
    }
    modules.clear();
  }
}
