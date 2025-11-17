import 'package:libtorchdart/libtorchdart.dart';

void main() {
  final conv = Conv2D.make(numInChannels: 32, numOutChannels: 32);
  print(conv);
}
