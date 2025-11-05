import 'package:libtorchdart/src/autoencoder/autoencoder.dart';
import 'package:libtorchdart/src/safetensor/safetensor.dart';

Future<void> main() async {
  final safeTensors = await SafeTensorsFile.load(
    'models/diffusion/v1-5-pruned-emaonly.safetensors',
  );
  final loader = safeTensors.mmapTensorLoader();
  final encoder = await VaeDecoder.loadFromSafeTensor(
    loader,
    prefix: 'first_stage_model.decoder.',
    // TODO
  );
  // TODO
}
