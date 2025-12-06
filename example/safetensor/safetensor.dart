import 'package:tensor/tensor.dart';

void main() async {
  final safeTensors = await SafeTensorsFile.load(
    './test_data/nn/conv2d/conv2d_tests.safetensors',
  );
  final tensorLoader = safeTensors.mmapTensorLoader();
  final input = tensorLoader.loadByName('test1.input');
  print('inputSize: ${input.sizes}');
}
