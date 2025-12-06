import 'package:tensor/src/tensor/tensor.dart';

void main() {
  final f32 = Tensor.arange(0, 10);
  print(f32.nativePtr);
  print(f32.dataType);
  final f64 = f32.to(dataType: DataType.float64);
  print(f64.nativePtr);
  print(f64.dataType);
}
