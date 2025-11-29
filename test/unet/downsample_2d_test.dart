import 'package:libtorchdart/libtorchdart.dart';
import 'package:test/test.dart';

void main() async {
  final context = Context.best();

  final tests = <_TestCase>[];
  final testDataFiles = [
    './test_data/unet/downsample/downsample_simple.safetensors',
    './test_data/unet/downsample/downsample_vae.safetensors',
  ];

  for (final fileName in testDataFiles) {
    final file = await SafeTensorsFile.load(fileName);
    final loader = file.mmapTensorLoader();
    final loadedTests = await _TestCase.loadAllFromSafeTensor(loader);
    tests.addAll(loadedTests);
  }

  group('DownSample2D', () {
    test('forward pass matches expected output', () {
      for (final test in tests) {
        final output = test.downsampler.forward(test.input, context: context);

        expect(output.shape, equals(test.output.shape));

        final result = test.output.allCloseSlow(output, atol: 1e-02);
        if (result != null) {
          print(
            'Max difference: $result, ${output.index(result)}, ${test.output.index(result)} ${output.index(result) - test.output.index(result)}',
          );
        }

        expect(
          test.output.allClose(output, atol: 1e-02),
          isTrue,
          reason:
              'Output tensor does not match expected output for ${test.name}',
        );
      }
    });
  });
}

class _TestCase {
  final String name;
  final Tensor input;
  final Tensor output;
  final DownSample2D downsampler;

  _TestCase({
    required this.name,
    required this.input,
    required this.output,
    required this.downsampler,
  });

  static Future<_TestCase> loadFromSafeTensor(
    SafeTensorLoader loader,
    String name,
  ) async {
    final input = await loader.loadByName('$name.input');
    final output = await loader.loadByName('$name.output');
    final padding = SymmetricPadding2D.fromPytorchString(
      loader.header.metadata['$name.padding']!,
    );

    final downsampler = await DownSample2D.loadFromSafeTensor(
      loader,
      prefix: '$name.downsample.',
      numChannels: input.shape[1], // Assuming NCHW layout
      padding: padding,
    );
    return _TestCase(
      name: name,
      input: input,
      output: output,
      downsampler: downsampler,
    );
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
