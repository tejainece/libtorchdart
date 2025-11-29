import 'package:libtorchdart/libtorchdart.dart';
import 'package:libtorchdart/src/autoencoder/encoder_block.dart';
import 'package:test/test.dart';

void main() async {
  final tests = <_TestCase>[];
  final testDataFiles = [
    './test_data/vae/down_encoder/down_encoder_simple.safetensors',
    './test_data/vae/down_encoder/down_encoder_vae.safetensors',
  ];

  for (final fileName in testDataFiles) {
    try {
      final file = await SafeTensorsFile.load(fileName);
      final loader = file.mmapTensorLoader();
      final loadedTests = await _TestCase.loadAllFromSafeTensor(loader);
      tests.addAll(loadedTests);
    } catch (e) {
      print('Warning: Could not load test file $fileName: $e');
    }
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
        final output = test.block.forward(test.input);

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

  _TestCase({
    required this.name,
    required this.input,
    required this.output,
    required this.block,
  });

  static Future<_TestCase> loadFromSafeTensor(
    SafeTensorLoader loader,
    String name,
  ) async {
    final input = await loader.loadByName('$name.input');
    final output = await loader.loadByName('$name.output');
    print(input.shape);
    final block = await DownEncoderBlock2D.loadFromSafeTensor(
      loader,
      prefix: '$name.block.',
    );
    return _TestCase(name: name, input: input, output: output, block: block);
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
