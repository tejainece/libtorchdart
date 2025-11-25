import 'package:libtorchdart/libtorchdart.dart';
import 'package:test/test.dart';

void main() async {
  Device device = Device(deviceType: DeviceType.cpu, deviceIndex: -1);

  final file = await SafeTensorsFile.load(
    './test_data/conv2d/conv2d_tests.safetensors',
  );
  final loader = file.mmapTensorLoader();
  final tests = await _TestCase.loadAllFromSafeTensor(loader);

  group('Conv2D', () {
    test('test', () {
      for (final test in tests) {
        final generator = Generator.getDefault(device: device);
        generator.currentSeed = 0;
        final conv = Conv2D.make(
          numInChannels: 32,
          numOutChannels: 32,
          kernelSize: SymmetricPadding2D.same(3),
          padding: SymmetricPadding2D.same(1),
          stride: SymmetricPadding2D.same(1),
          generator: generator,
          device: device,
        );
        final output = conv.forward(test.input);
        expect(test.output.allClose(output), true);
        print('success');
      }
    });
  });
}

class _TestCase {
  final Tensor input;
  final Tensor output;

  _TestCase({required this.input, required this.output});

  static Future<_TestCase> loadFromSafeTensor(
    SafeTensorLoader loader,
    String name,
  ) async {
    final output = await loader.loadByName('$name.output');
    final input = await loader.loadByName('$name.input');
    return _TestCase(input: input, output: output);
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
