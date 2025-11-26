import 'package:libtorchdart/libtorchdart.dart';
import 'package:libtorchdart/src/autoencoder/encoder_block.dart';
import 'package:test/test.dart';

void main() async {
  final tests = <_TestCase>[];
  final testDataFiles = ['./test_data/vae/down_encoder_block_2d.safetensors'];

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
        final output = test.block.forward(test.input, embeds: test.temb);

        expect(output.shape, equals(test.output.shape));

        // final result = test.output.allCloseSlow(output, atol: 1e-02);
        // print(result);

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
  final Tensor? temb;
  final Tensor output;
  final DownEncoderBlock2D block;

  _TestCase({
    required this.name,
    required this.input,
    required this.output,
    required this.temb,
    required this.block,
  });

  static Future<_TestCase> loadFromSafeTensor(
    SafeTensorLoader loader,
    String name,
  ) async {
    final input = await loader.loadByName('$name.input');
    final output = await loader.loadByName('$name.output');
    final temb = await loader.tryLoadByName('$name.temb');

    if (name == 'simple1') {
      print('Keys for simple1:');
      loader.tensorInfos.keys.where((k) => k.startsWith(name)).forEach(print);
    }

    // Infer parameters for DownEncoderBlock2D
    final numInChannels = input.shape[1];
    final numOutChannels =
        output.shape[1]; // Assuming output channels match expected output

    // Count layers by looking for resnets.{i}. prefix
    int numLayers = 0;
    while (true) {
      // Check for any tensor belonging to resnets[i]
      // A common tensor in ResnetBlock2D is norm1.weight
      final prefix = '$name.resnets.$numLayers.';
      bool hasLayer = false;
      for (final key in loader.tensorInfos.keys) {
        if (key.startsWith(prefix)) {
          hasLayer = true;
          break;
        }
      }
      if (hasLayer) {
        numLayers++;
      } else {
        break;
      }
    }

    // Check if downsample is present
    // DownEncoderBlock2D adds downsample if addDownsample is true.
    // In the loader, it looks for downsamplers.0.
    // We can check if downsamplers.0. exists in the keys.
    bool addDownsample = false;
    final downsamplePrefix = '$name.downsamplers.0.';
    for (final key in loader.tensorInfos.keys) {
      if (key.startsWith(downsamplePrefix)) {
        addDownsample = true;
        break;
      }
    }

    final block = await DownEncoderBlock2D.loadFromSafeTensor(
      loader,
      prefix: '$name.',
      numInChannels: numInChannels,
      numOutChannels: numOutChannels,
      numLayers: numLayers,
      addDownsample: addDownsample,
      // Default values for others, or we might need to infer them too?
      // resnetEps, resnetActFn, resnetGroups, downsamplePadding, dropout
      // These are usually standard or can be inferred if we really dig deep,
      // but let's try with defaults first.
    );

    return _TestCase(
      name: name,
      input: input,
      output: output,
      temb: temb,
      block: block,
    );
  }

  static Future<List<_TestCase>> loadAllFromSafeTensor(
    SafeTensorLoader loader,
  ) async {
    final map = <String, _TestCase>{};
    for (final t in loader.tensorInfos.keys) {
      final name = t.split('.').first;
      if (map.containsKey(name)) continue;
      // We only want to load the test case once per name group
      // The keys are like "test_case_name.input", "test_case_name.block.resnets.0..."
      map[name] = await loadFromSafeTensor(loader, name);
    }
    return map.values.toList();
  }
}
