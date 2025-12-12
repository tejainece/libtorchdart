import 'package:test/test.dart';
import 'package:tensor/tensor.dart';

void main() {
  group('FInfo', () {
    test('float32', () {
      final info = DataType.float32.fInfo;
      expect(info.min, lessThan(0));
      expect(info.max, greaterThan(0));
      expect(info.eps, greaterThan(0));
      expect(info.tiny, greaterThan(0));
      expect(info.resolution, isNotNull);
    });

    test('float64', () {
      final info = DataType.float64.fInfo;
      expect(info.min, lessThan(0));
      expect(info.max, greaterThan(0));
      expect(info.eps, greaterThan(0));
      expect(info.tiny, greaterThan(0));
      expect(info.resolution, isNotNull);
    });

    test('non-floating point throws', () {
      expect(() => DataType.int64.fInfo, throwsArgumentError);
    });
  });
}
