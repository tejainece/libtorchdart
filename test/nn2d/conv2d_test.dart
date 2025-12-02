import 'package:libtorchdart/libtorchdart.dart';
import 'package:test/test.dart';

void main() async {
  final context = Context.best();

  final file = await SafeTensorsFile.load(
    './test_data/conv2d/conv2d_simple.safetensors',
  );
  final loader = file.mmapTensorLoader();
  final tests = await _TestCase.loadAllFromSafeTensor(
    loader,
    device: context.device,
  );

  group('Conv2D', () {
    test('test', () {
      for (final test in tests) {
        print(test.name);
        final output = test.conv.forward(test.input, context: context);
        expect(test.output.allClose(output), true);
      }
      print('success');
    });
  });
}

class _TestCase {
  final String name;
  final Tensor input;
  final Tensor output;
  final Conv2D conv;

  _TestCase({
    required this.name,
    required this.input,
    required this.output,
    required this.conv,
  });

  static Future<_TestCase> loadFromSafeTensor(
    SafeTensorLoader loader,
    String name, {
    required Device device,
  }) async {
    final output = await loader.loadByName('$name.output', device: device);
    final input = await loader.loadByName('$name.input', device: device);

    final padding = SymmetricPadding2D.fromPytorchString(
      loader.header.metadata['$name.padding']!,
    );
    final stride = SymmetricPadding2D.fromPytorchString(
      loader.header.metadata['$name.stride']!,
    );
    final dilation = SymmetricPadding2D.fromPytorchString(
      loader.header.metadata['$name.dilation']!,
    );
    final groups = int.parse(loader.header.metadata['$name.groups']!);
    PadMode? paddingMode = PadMode.tryFromPytorchString(
      loader.header.metadata['$name.padding_mode'],
    );
    if (paddingMode == PadMode.constant) {
      paddingMode = null;
    }
    final conv = await Conv2D.loadFromSafeTensor(
      loader,
      prefix: '$name.conv.',
      padding: padding,
      stride: stride,
      dilation: dilation,
      groups: groups,
      padMode: paddingMode,
    );
    return _TestCase(name: name, input: input, output: output, conv: conv);
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
