import 'package:libtorchdart/libtorchdart.dart';
import 'package:libtorchdart/src/autoencoder/encoder_block.dart';
import 'package:test/test.dart';

void main() async {
  final context = Context.best();

  final tests = <_TestCase>[];
  final testDataFiles = [
    './test_data/vae/down_encoder/downencoder_simple.safetensors',
    './test_data/vae/down_encoder/downencoder_vae.safetensors',
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
        final output = test.block.forward(test.input, context: context);

        expect(output.shape, equals(test.output.shape));
        /*final result = test.output.allCloseSlow(output, atol: 1e-02);
        print(result);*/

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
    String name, {
    required Device device,
  }) async {
    final input = await loader.loadByName('$name.input', device: device);
    final output = await loader.loadByName('$name.output', device: device);
    final resnetEps = double.parse(loader.header.metadata['$name.resnet_eps']!);
    final activation = Activation.fromName(
      loader.header.metadata['$name.resnet_act_fn']!,
    )!;
    final downsamplePadding = SymmetricPadding2D.fromPytorchString(
      loader.header.metadata['$name.downsample_padding']!,
    );
    final block = await DownEncoderBlock2D.loadFromSafeTensor(
      loader,
      prefix: '$name.block.',
      resnetEps: resnetEps,
      resnetActFn: activation,
      resnetGroups: int.parse(loader.header.metadata['$name.resnet_groups']!),
      downsamplePadding: downsamplePadding,
      outputScaleFactor: double.parse(
        loader.header.metadata['$name.output_scale_factor']!,
      ),
    );

    return _TestCase(name: name, input: input, output: output, block: block);
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
