import 'package:libtorchdart/libtorchdart.dart';
import 'package:test/test.dart';

class _TestCase {
  final String name;
  final Tensor input;
  final Tensor output;
  final Conv2DTranspose conv;

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
    final outputPadding = SymmetricPadding2D.fromPytorchString(
      loader.header.metadata['$name.output_padding']!,
    );
    final conv = await Conv2DTranspose.loadFromSafeTensor(
      loader,
      prefix: '$name.conv.',
      padding: padding,
      stride: stride,
      dilation: dilation,
      groups: groups,
      outputPadding: outputPadding,
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

void main() async {
  final context = Context.best();

  final file = await SafeTensorsFile.load(
    './test_data/nn2d/conv2d_transpose/simple.safetensors',
  );
  final loader = file.mmapTensorLoader();
  final tests = await _TestCase.loadAllFromSafeTensor(
    loader,
    device: context.device,
  );

  group('Conv2DTranspose.loadFromSafeTensor', () {
    test('test', () {
      for (final test in tests) {
        print(test.name);
        final output = test.conv.forward(test.input, context: context);

        /*final result = test.output.allCloseSlow(output, atol: 1e-03);
        print(result);*/

        expect(test.output.allClose(output, atol: 1e-03), true);
      }
      print('success');
    });
  });

  group('Conv2DTransposed.make', () {
    test('make creates Conv2DTransposed with correct dimensions', () {
      final conv = Conv2DTranspose.make(
        numInChannels: 32,
        numOutChannels: 64,
        kernelSize: SymmetricPadding2D.same(3),
      );

      expect(conv.numInChannels, equals(32));
      expect(conv.numOutChannels, equals(64));
      expect(conv.kernelSize, equals(SymmetricPadding2D.same(3)));
      expect(conv.weight.shape, equals([32, 64, 3, 3]));
      expect(conv.bias, isNotNull);
      expect(conv.bias!.shape, equals([64]));
    });

    /*
    // TODO: Fix grouped convolution test - need to verify correct weight dimensions
    // test('make with groups creates Conv2DTransposed with correct dimensions', () {
    //   final conv = Conv2DTransposed.make(
    //     numInChannels: 32,
    //     numOutChannels: 64,
    //     kernelSize: SymmetricPadding2D.same(3),
    //     groups: 2,
    //   );
    //
    //   expect(conv.numInChannels, equals(32));
    //   expect(conv.numOutChannels, equals(64));
    //   expect(conv.groups, equals(2));
    //   expect(conv.weight.shape, equals([32, 32, 3, 3]));
    // });*/

    test('make without bias creates Conv2DTransposed without bias', () {
      final conv = Conv2DTranspose.make(
        numInChannels: 16,
        numOutChannels: 32,
        hasBias: false,
      );

      expect(conv.bias, isNull);
    });

    test('forward pass with stride=2 increases spatial dimensions', () {
      final conv = Conv2DTranspose.make(
        numInChannels: 32,
        numOutChannels: 16,
        kernelSize: SymmetricPadding2D.same(4),
        stride: SymmetricPadding2D.same(2),
        padding: SymmetricPadding2D.same(1),
        device: context.device,
      );

      final input = Tensor.ones([1, 32, 8, 8], device: context.device);
      final output = conv.forward(input, context: context);

      // Output size formula: (input_size - 1) * stride - 2 * padding + kernel_size + output_padding
      // (8 - 1) * 2 - 2 * 1 + 4 + 0 = 14 - 2 + 4 = 16
      expect(output.shape, equals([1, 16, 16, 16]));
    });

    test('forward pass with kernel_size=3, stride=1 works correctly', () {
      final conv = Conv2DTranspose.make(
        numInChannels: 16,
        numOutChannels: 32,
        kernelSize: SymmetricPadding2D.same(3),
        stride: SymmetricPadding2D.same(1),
        padding: SymmetricPadding2D.same(1),
        device: context.device,
      );

      final input = Tensor.ones([2, 16, 14, 14], device: context.device);
      final output = conv.forward(input, context: context);

      // Output size: (14 - 1) * 1 - 2 * 1 + 3 + 0 = 13 - 2 + 3 = 14
      expect(output.shape, equals([2, 32, 14, 14]));
    });

    test('forward pass with output_padding adds extra output size', () {
      final conv = Conv2DTranspose.make(
        numInChannels: 16,
        numOutChannels: 16,
        kernelSize: SymmetricPadding2D.same(3),
        stride: SymmetricPadding2D.same(2),
        padding: SymmetricPadding2D.same(1),
        outputPadding: SymmetricPadding2D.same(1),
        device: context.device,
      );

      final input = Tensor.ones([1, 16, 7, 7], device: context.device);
      final output = conv.forward(input, context: context);

      // Output size: (7 - 1) * 2 - 2 * 1 + 3 + 1 = 12 - 2 + 3 + 1 = 14
      expect(output.shape, equals([1, 16, 14, 14]));
    });

    test('forward pass with non-square kernel works', () {
      final conv = Conv2DTranspose.make(
        numInChannels: 8,
        numOutChannels: 16,
        kernelSize: SymmetricPadding2D(vertical: 3, horizontal: 5),
        stride: SymmetricPadding2D.same(1),
        padding: SymmetricPadding2D.same(0),
      );

      final input = Tensor.ones([1, 8, 10, 10], device: context.device);
      final output = conv.forward(input, context: context);

      // Output height: (10 - 1) * 1 - 2 * 0 + 3 = 12
      // Output width: (10 - 1) * 1 - 2 * 0 + 5 = 14
      expect(output.shape, equals([1, 16, 12, 14]));
    });

    test('forward pass with asymmetric stride works', () {
      final conv = Conv2DTranspose.make(
        numInChannels: 8,
        numOutChannels: 16,
        kernelSize: SymmetricPadding2D.same(3),
        stride: SymmetricPadding2D(vertical: 1, horizontal: 2),
        padding: SymmetricPadding2D.same(1),
      );

      final input = Tensor.ones([1, 8, 10, 10], device: context.device);
      final output = conv.forward(input, context: context);

      // Output height: (10 - 1) * 1 - 2 * 1 + 3 = 10
      // Output width: (10 - 1) * 2 - 2 * 1 + 3 = 19
      expect(output.shape, equals([1, 16, 10, 19]));
    });

    test('weight dimensions follow PyTorch convention', () {
      final conv = Conv2DTranspose.make(
        numInChannels: 64,
        numOutChannels: 32,
        kernelSize: SymmetricPadding2D.same(4),
        groups: 1,
        device: context.device,
      );

      // For transposed conv: [in_channels, out_channels/groups, kH, kW]
      expect(conv.weight.shape[0], equals(64)); // in_channels
      expect(conv.weight.shape[1], equals(32)); // out_channels
      expect(conv.weight.shape[2], equals(4)); // kernel height
      expect(conv.weight.shape[3], equals(4)); // kernel width
    });
  });
}
