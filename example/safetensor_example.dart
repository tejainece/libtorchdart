import 'package:libtorchdart/src/safetensor/safetensor.dart';

void main() async {
  final tensor = await RandomAccessSafeTensors.load('data/safetensors/SophraxiaChroma-000012.safetensors');
  print(tensor.header.metadata.keys);
}