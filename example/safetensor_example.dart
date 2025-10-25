import 'package:libtorchdart/libtorchdart.dart';
import 'package:libtorchdart/src/safetensor/storage.dart';

void main() async {
  // print(DataType.fromSafeTensorName('BF16'));
  final safeTensors = await SafeTensorsFile.load(
    'models/diffusion/v1-5-pruned-emaonly.safetensors',
  );
  final tensorLoader = safeTensors.mmapTensorLoader();
  for (final tensorInfoEntry in tensorLoader.tensorInfos.entries) {
    final tensorInfo = tensorInfoEntry.value;
    print(
      '${tensorInfoEntry.key} shape: ${tensorInfo.shape} ${tensorInfo.bytes} ${tensorInfo.startOffset} ${tensorInfo.endOffset}',
    );
  }

  await printTensor(tensorLoader, 'model_ema.decay');
  await printTensor(
    tensorLoader,
    'model.diffusion_model.output_blocks.9.1.transformer_blocks.0.norm1.bias',
  );

  tensorLoader.release();
  print('Finished!');
}

Future<void> printTensor(SafeTensorLoader tensorLoader, String name) async {
  print('++++++++++++$name+++++++++++');
  final tensorInfo = tensorLoader.tensorInfos[name]!;
  print(
    '${tensorInfo.bytes} ${tensorInfo.startOffset} ${tensorInfo.endOffset}',
  );
  final tensor = await tensorLoader.loadByName(name);
  print('shape: ${tensor.shape}');
  print('dim: ${tensor.dim}');
  print('device: ${tensor.device}');
  if (tensor.isScalar) {
    print('scalar: ${tensor.scalar}');
  }
  print('-----------$name------------');
}
