import 'dart:ffi';
import 'package:ffi/ffi.dart';
import 'package:tensor/tensor.dart';
import 'package:test/test.dart';

void main() {
  group('Tensor.fromBlob', () {
    test('creates tensor from raw pointer (Float64)', () {
      final size = 10;
      final ptr = calloc<Double>(size);
      for (var i = 0; i < size; i++) {
        ptr[i] = i.toDouble();
      }

      final tensor = Tensor.fromBlob(ptr.cast(), [
        size,
      ], datatype: DataType.float64);

      expect(tensor.shape, [size]);
      expect(tensor.dataType, DataType.float64);
      expect(tensor.dataPointer, ptr);
      for (var i = 0; i < size; i++) {
        expect(tensor.scalarAt(i), i.toDouble());
      }

      // Modify memory, check tensor
      ptr[0] = 100.0;
      expect(tensor.scalarAt(0), 100.0);

      // Modify tensor, check memory
      // tensor.get(1) returns a view of the element at index 1
      tensor.get(1).fill_(200.0);
      expect(ptr[1], 200.0);

      // Clean up
      calloc.free(ptr);
    });

    test('creates tensor from raw pointer (Float32)', () {
      final size = 5;
      final ptr = calloc<Float>(size);
      for (var i = 0; i < size; i++) {
        ptr[i] = i.toDouble();
      }

      final tensor = Tensor.fromBlob(ptr.cast(), [
        size,
      ], datatype: DataType.float32);

      expect(tensor.shape, [size]);
      expect(tensor.dataType, DataType.float32);
      expect(tensor.dataPointer, ptr);
      for (var i = 0; i < size; i++) {
        // Float32 precision might be an issue, but for small integers it should be fine
        expect(tensor.scalarAt(i), i.toDouble());
      }

      // Modify memory
      ptr[0] = 50.0;
      expect(tensor.scalarAt(0), 50.0);

      // Modify tensor
      tensor.get(1).fill_(200.0);
      expect(ptr[1], 200.0);

      calloc.free(ptr);
    });

    test('creates tensor from raw pointer (Int64)', () {
      final size = 5;
      final ptr = calloc<Int64>(size);
      for (var i = 0; i < size; i++) {
        ptr[i] = i;
      }

      final tensor = Tensor.fromBlob(ptr.cast(), [
        size,
      ], datatype: DataType.int64);

      expect(tensor.shape, [size]);
      expect(tensor.dataType, DataType.int64);
      expect(tensor.dataPointer, ptr);
      for (var i = 0; i < size; i++) {
        expect(tensor.scalarAt(i), i);
      }

      ptr[0] = 99;
      expect(tensor.scalarAt(0), 99);

      // Modify tensor
      tensor.get(1).fill_(200);
      expect(ptr[1], 200);

      calloc.free(ptr);
    });

    test('creates tensor from raw pointer (Int32)', () {
      final size = 5;
      final ptr = calloc<Int32>(size);
      for (var i = 0; i < size; i++) {
        ptr[i] = i;
      }

      final tensor = Tensor.fromBlob(ptr.cast(), [
        size,
      ], datatype: DataType.int32);

      expect(tensor.shape, [size]);
      expect(tensor.dataType, DataType.int32);
      expect(tensor.dataPointer, ptr);
      for (var i = 0; i < size; i++) {
        expect(tensor.scalarAt(i), i);
      }

      ptr[0] = 99;
      expect(tensor.scalarAt(0), 99);

      // Modify tensor
      tensor.get(1).fill_(200);
      expect(ptr[1], 200);

      calloc.free(ptr);
    });
  });
}
