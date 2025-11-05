import 'package:libtorchdart/libtorchdart.dart';

void main() async {
  // print(DataType.fromSafeTensorName('BF16'));
  final safeTensors = await SafeTensorsFile.load(
    './models/diffusion/v1-5-pruned-emaonly-fp16.safetensors',
  );
}
