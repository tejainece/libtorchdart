import 'package:libtorchdart/libtorchdart.dart';
import 'package:libtorchdart/src/autoencoder/encoder_block.dart';
import 'package:test/test.dart';

void main() async {
  final context = Context.best();

  final tests = <_TestCase>[];
  final testDataFiles = [
    './test_data/vae/down_encoder/down_encoder_simple.safetensors',
    // TODO './test_data/vae/down_encoder/down_encoder_vae.safetensors',
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

  group('DownEncoderBlock2D', () {
    test('forward pass matches expected output', () {
      if (tests.isEmpty) {
        print('No tests loaded. Skipping.');
        return;
      }

      for (final test in tests) {
        print('Running test: ${test.name}');
        print('Block has ${test.block.resnets.length} resnets');
        for (var i = 0; i < test.block.resnets.length; i++) {
          final r = test.block.resnets[i];
          print(
            'Resnet $i: ${r.conv1.numInChannels}→${r.conv1.numOutChannels}, ${r.conv2.numInChannels}→${r.conv2.numOutChannels}${r.convShortcut != null ? ", shortcut: ${r.convShortcut!.numInChannels}→${r.convShortcut!.numOutChannels}" : ""}',
          );
        }
        final output = test.block.forward(test.input, context: context);

        expect(output.shape, equals(test.output.shape));
        final result = test.output.allCloseSlow(output, atol: 1e-02);
        print(result);

        expect(
          test.output.allClose(output, atol: 1e-02),
          isTrue,
          reason:
              'Output tensor does not match expected output for ${test.name}',
        );
        print('Test ${test.name} passed');
      }
    });
  });
}

class _TestCase {
  final String name;
  final Tensor input;
  final Tensor output;
  final DownEncoderBlock2D block;
  final String downSamplerPrefix;
  final String resnetPrefix;

  _TestCase({
    required this.name,
    required this.input,
    required this.output,
    required this.block,
    required this.downSamplerPrefix,
    required this.resnetPrefix,
  });

  static Future<_TestCase> loadFromSafeTensor(
    SafeTensorLoader loader,
    String name, {
    required Device device,
  }) async {
    final input = await loader.loadByName('$name.input', device: device);
    final output = await loader.loadByName('$name.output', device: device);
    final block = await DownEncoderBlock2D.loadFromSafeTensor(
      loader,
      prefix: '$name.block.',
    );
    final downSamplerPrefix = 'blockdownsamplers.';
    final resnetPrefix = 'blockresnets.';

    return _TestCase(
      name: name,
      input: input,
      output: output,
      block: block,
      downSamplerPrefix: downSamplerPrefix,
      resnetPrefix: resnetPrefix,
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
