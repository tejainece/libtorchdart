import 'package:libtorchdart/libtorchdart.dart';

void main() {
  final context = Context.best();
  print('Using device: ${context.device}');

  // Create a Conv2DTransposed layer for upsampling
  final conv = ConvTranspose2D.make(
    numInChannels: 64,
    numOutChannels: 32,
    kernelSize: SymmetricPadding2D.same(4),
    stride: SymmetricPadding2D.same(2),
    padding: SymmetricPadding2D.same(1),
    device: context.device,
  );

  print('\nConv2DTransposed layer:');
  print(conv);

  // Create input tensor [batch, channels, height, width]
  final input = Tensor.randn([1, 64, 16, 16], device: context.device);
  print('\nInput shape: ${input.shape}');

  // Forward pass - doubles spatial dimensions
  final output = conv.forward(input, context: context);
  print('Output shape: ${output.shape}');

  print('\nWeight shape: ${conv.weight.shape}');
  if (conv.bias != null) {
    print('Bias shape: ${conv.bias!.shape}');
  }
}
