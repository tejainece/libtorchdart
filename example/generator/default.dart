import 'package:libtorchdart/libtorchdart.dart';

void main() {
  final generator = Generator.getDefault();
  print(generator.currentSeed);
  generator.currentSeed = 0;
  print(generator.currentSeed);
}
