import 'package:libtorchdart/libtorchdart.dart';
import 'package:libtorchdart/src/transformers/clip_text/clip_config.dart';

Future<void> main() async {
  final prompt = 'minimalistic symmetrical logo with moose head';
  final clip = await CLIPTokenizer.loadFromFile(
    'models/diffusion/bpe_simple_vocab_16e6.txt',
    config: ClipTextConfig.v2_1,
  );
  final tokens = clip.encode(prompt);
  print(tokens);
  final decoded = clip.decode(tokens);
  print(decoded);
}
