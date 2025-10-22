import 'package:libtorchdart/libtorchdart.dart';

Future<void> main() async {
  final clip = await CLIPTokenizer.loadFromFile('data/bpe_simple_vocab_16e6.txt', config: CLIPConfig.v2_1);
  //print(clip.bpeEncode('hello'));
  print(clip.encodeWithPad('how are you, unbelievably beautiful'));
  // TODO
}