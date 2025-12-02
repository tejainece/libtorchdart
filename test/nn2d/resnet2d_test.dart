import 'package:libtorchdart/libtorchdart.dart';
import 'package:test/test.dart';

void main() async {
  final context = Context.best();

  final tests = <_TestCase>[];
  final testDataFiles = [
    './test_data/resnet/resnet_tests.safetensors',
    './test_data/resnet/resnet_sd15_unet_tests.safetensors',
    './test_data/resnet/resnet_sd15_vae_tests.safetensors',
  ];
  for (final fileName in testDataFiles) {
    final file = await SafeTensorsFile.load(fileName);
    final loader = file.mmapTensorLoader();
    final loadedTests = await _TestCase.loadAllFromSafeTensor(
      loader,
      device: context.device,
    );
    tests.addAll(loadedTests);
  }

  group('ResnetBlock2D', () {
    test('basic block with time embedding', () {
      for (final test in tests) {
        // print(test.resnet);
        final output = test.resnet.forward(
          test.input,
          embeds: test.temb,
          context: context,
        );

        expect(output.shape, equals(test.output.shape));

        final result = test.output.allCloseSlow(output, atol: 1e-02);
        print(result);
        expect(
          test.output.allClose(output, atol: 1e-02),
          isTrue,
          reason: 'Output tensor does not match expected output',
        );
        print('Test ${test.name} passed');
      }
    });
  });
}

class _TestCase {
  final String name;
  final Tensor input;
  final Tensor? temb;
  final Tensor output;
  final ResnetBlock2D resnet;

  _TestCase({
    required this.name,
    required this.input,
    required this.output,
    required this.temb,
    required this.resnet,
  });

  static Future<_TestCase> loadFromSafeTensor(
    SafeTensorLoader loader,
    String name, {
    required Device device,
  }) async {
    final input = await loader.loadByName('$name.input', device: device);
    final output = await loader.loadByName('$name.output', device: device);
    final temb = await loader.tryLoadByName('$name.temb', device: device);
    final resnet = await ResnetBlock2D.loadFromSafeTensor(
      loader,
      prefix: '$name.resnet.',
    );
    return _TestCase(
      name: name,
      input: input,
      output: output,
      temb: temb,
      resnet: resnet,
    );
  }

  static Future<List<_TestCase>> loadAllFromSafeTensor(
    SafeTensorLoader loader, {
    required Device device,
  }) async {
    final map = <String, _TestCase>{};
    for (final t in loader.tensorInfos.keys) {
      final name = t.split('.').first;
      if (map.containsKey(name)) continue;
      map[name] = await loadFromSafeTensor(loader, name, device: device);
    }
    return map.values.toList();
  }
}
