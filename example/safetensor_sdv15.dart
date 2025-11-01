import 'package:libtorchdart/libtorchdart.dart';

void main() async {
  // print(DataType.fromSafeTensorName('BF16'));
  final safeTensors = await SafeTensorsFile.load(
    '/home/tejag/comfy/ComfyUI/models/checkpoints/v1-5-pruned-emaonly-fp16.safetensors',
  );
}
