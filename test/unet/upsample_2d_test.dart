import 'package:libtorchdart/libtorchdart.dart';
import 'package:test/test.dart';

void main() async {
  final context = Context.best();

  final tests = <_TestCase>[];
  final testDataFiles = [
    './test_data/unet/upsample/upsample_simple.safetensors',
    './test_data/unet/upsample/upsample_vae.safetensors',
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

  group('UpSample2D', () {
    test('forward pass matches expected output', () {
      for (final test in tests) {
        final output = test.upsampler.forward(
          test.input,
          outputSize: SymmetricPadding2D(
            vertical: test.output.shape[2],
            horizontal: test.output.shape[3],
          ),
          context: context,
        );

        expect(output.shape, equals(test.output.shape));

        /*
        final result = test.output.allCloseSlow(output, atol: 1e-02);
        if (result != null) {
          print(
            'Max difference: $result, ${output.index(result)}, ${test.output.index(result)} ${output.index(result) - test.output.index(result)}',
          );
        }*/

        expect(
          test.output.allClose(output, atol: 1e-02),
          isTrue,
          reason:
              'Output tensor does not match expected output for ${test.name}',
        );
      }
    });

    test('make with useConv=false should have null conv', () {
      final upsampler = Upsample2D.make(numChannels: 32, useConv: false);
      expect(upsampler.conv, isNull);
    });

    test('make with useConv=true should have conv', () {
      final upsampler = Upsample2D.make(numChannels: 32, useConv: true);
      expect(upsampler.conv, isNotNull);
      expect(upsampler.conv!.numInChannels, equals(32));
    });
  });
}

class _TestCase {
  final String name;
  final Tensor input;
  final Tensor output;
  final Upsample2D upsampler;

  _TestCase({
    required this.name,
    required this.input,
    required this.output,
    required this.upsampler,
  });

  static Future<_TestCase> loadFromSafeTensor(
    SafeTensorLoader loader,
    String name, {
    required Device device,
  }) async {
    final input = await loader.loadByName('$name.input', device: device);
    final output = await loader.loadByName('$name.output', device: device);
    final padding = SymmetricPadding2D.fromPytorchString(
      loader.header.metadata['$name.padding']!,
    );

    final upsampler = await Upsample2D.loadFromSafeTensor(
      loader,
      prefix: '$name.upsample.',
      numChannels: input.shape[1], // Assuming NCHW layout
      padding: padding,
    );
    return _TestCase(
      name: name,
      input: input,
      output: output,
      upsampler: upsampler,
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
