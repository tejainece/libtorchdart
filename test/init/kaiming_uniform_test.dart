import 'dart:math';

import 'package:tensor/tensor.dart';
import 'package:test/expect.dart';
import 'package:test/scaffolding.dart';

Future<void> main() async {
  Device device = Device(deviceType: DeviceType.cpu, deviceIndex: -1);

  final file = await SafeTensorsFile.load(
    './test_data/init/kaiming_uniform_tests.safetensors',
  );
  final loader = file.cpuLoader();
  final tests = await _TestCase.loadAllFromSafeTensor(loader);

  group('Init.KaimingUniform_', () {
    test('tests', () {
      for (final test in tests) {
        final generator = Generator.getDefault(device: device);
        generator.currentSeed = test.seed;
        final weights = Tensor.empty(test.size, device: device);
        Init.kaimingUniform_(weights, a: sqrt(5), generator: generator);
        expect(
          test.output.allClose(weights),
          true,
          reason: 'for ${test.name}; output mismatch',
        );
      }
    });
  });
}

class _TestCase {
  final String name;
  final Tensor output;
  final int seed;

  _TestCase({required this.name, required this.output, required this.seed});

  late final List<int> size = output.sizes;

  static Future<_TestCase> loadFromSafeTensor(
    SafeTensorLoader loader,
    String name,
  ) async {
    final output = await loader.loadByName('$name.output');
    final int seed = ((await loader.loadByName('$name.seed')).scalar as num)
        .toInt();
    return _TestCase(name: name, output: output, seed: seed);
  }

  static Future<List<_TestCase>> loadAllFromSafeTensor(
    SafeTensorLoader loader,
  ) async {
    final map = <String, _TestCase>{};
    for (final t in loader.tensorInfos.keys) {
      final name = t.split('.').first;
      if (map.containsKey(name)) continue;
      map[name] = await loadFromSafeTensor(loader, name);
    }
    return map.values.toList();
  }
}
