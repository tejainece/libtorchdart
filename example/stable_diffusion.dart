import 'package:libtorchdart/libtorchdart.dart';

Future<void> main() async {
  final prompt = 'minimalistic symmetrical logo with moose head';
  final negativePrompt = '';
  final clip = await CLIPTokenizer.loadFromFile('data/bpe_simple_vocab_16e6.txt', config: CLIPConfig.v2_1);
  final tokens = clip.encode(prompt);
  final uncondTokens = clip.encode(negativePrompt);
  print(tokens);
  print(uncondTokens);
  // TODO tokens to tensor
  // TODO
}