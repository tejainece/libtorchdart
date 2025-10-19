import 'package:libtorchdart/libtorchdart.dart';

void main() async {
  // print(DataType.fromSafeTensorName('BF16'));
  final safeTensors = await SafeTensorsFile.load(
    'data/safetensors/SophraxiaChroma-000012.safetensors',
  );
  final tensorLoader = safeTensors.mmapTensorLoader();
  for (final tensorInfoEntry in tensorLoader.tensorInfos.entries) {
    final tensorInfo = tensorInfoEntry.value;
    print(
      '${tensorInfoEntry.key} ${tensorInfo.bytes} ${tensorInfo.startOffset} ${tensorInfo.endOffset}',
    );
  }
  final tensorInfo = tensorLoader
      .tensorInfos['lora_unet_single_blocks_31_linear2.lora_down.weight']!;
  print(
    '${tensorInfo.bytes} ${tensorInfo.startOffset} ${tensorInfo.endOffset}',
  );
  final tensor = tensorLoader.loadByName(
    'lora_unet_single_blocks_31_linear2.lora_down.weight',
  );
  print('shape: ${tensor.shape}');
  print('dim: ${tensor.dim}');
  print('device: ${tensor.device}');
  if (tensor.isScalar) {
    print('scalar: ${tensor.scalar}');
  }
  final tensor1 = tensor.get(0).get(15359);
  print('shape: ${tensor1.shape} ${tensor1.scalar}');
  tensorLoader.release();
  print('Finished!');
}
