import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart' as ffi;
import 'package:libtorchdart/src/tensor_ffi/tensor_ffi.dart';

class Tensor {
  ffi.Pointer<ffi.Void> nativePtr;

  Tensor(this.nativePtr);

  static Tensor zeros(
    List<int> sizes, {
    Device device = Device.cpu,
    DataType dtype = DataType.float,
    Layout layout = Layout.strided,
    // TODO memory format
    // TODO autograd
    // TODO pinned memory
  }) {
    final arena = ffi.Arena();
    try {
      final options = FFITensorOptions.make(
        dataType: dtype,
        device: device,
        layout: layout,
        allocator: arena,
      );
      final sizesPointer = arena.allocate<ffi.Int64>(
        ffi.sizeOf<ffi.Int64>() * sizes.length,
      );
      sizesPointer.asTypedList(sizes.length).setAll(0, sizes);
      final tensor = TensorFFI.zeros(sizesPointer, sizes.length, options.ref);
      return Tensor(tensor);
    } finally {
      arena.releaseAll();
    }
  }

  static Tensor ones(
    List<int> sizes, {
    Device device = Device.cpu,
    DataType dtype = DataType.float,
    Layout layout = Layout.strided,
    // TODO memory format
    // TODO autograd
    // TODO pinned memory
  }) {
    final arena = ffi.Arena();
    try {
      final options = FFITensorOptions.make(
        dataType: dtype,
        device: device,
        layout: layout,
        allocator: arena,
      );
      final sizesPointer = arena.allocate<ffi.Int64>(
        ffi.sizeOf<ffi.Int64>() * sizes.length,
      );
      sizesPointer.asTypedList(sizes.length).setAll(0, sizes);
      final tensor = TensorFFI.ones(sizesPointer, sizes.length, options.ref);
      return Tensor(tensor);
    } finally {
      arena.releaseAll();
    }
  }

  static Tensor arange(
    int end, {
    Device device = Device.cpu,
    DataType dtype = DataType.float,
    Layout layout = Layout.strided,
    // TODO memory format
    // TODO autograd
    // TODO pinned memory
  }) {
    final arena = ffi.Arena();
    try {
      final options = FFITensorOptions.make(
        dataType: dtype,
        device: device,
        layout: layout,
        allocator: arena,
      );
      final tensor = TensorFFI.arange(end, options.ref);
      return Tensor(tensor);
    } finally {
      arena.releaseAll();
    }
  }

  static Tensor rand(
    List<int> sizes, {
    Device device = Device.cpu,
    DataType dtype = DataType.float,
    Layout layout = Layout.strided,
    // TODO memory format
    // TODO autograd
    // TODO pinned memory
  }) {
    final arena = ffi.Arena();
    try {
      final options = FFITensorOptions.make(
        dataType: dtype,
        device: device,
        layout: layout,
        allocator: arena,
      );
      final sizesPointer = arena.allocate<ffi.Int64>(
        ffi.sizeOf<ffi.Int64>() * sizes.length,
      );
      sizesPointer.asTypedList(sizes.length).setAll(0, sizes);
      final tensor = TensorFFI.rand(sizesPointer, sizes.length, options.ref);
      return Tensor(tensor);
    } finally {
      arena.releaseAll();
    }
  }

  static Tensor eye(
    int n, {
    int? m,
    Device device = Device.cpu,
    DataType dtype = DataType.float,
    Layout layout = Layout.strided,
    // TODO memory format
    // TODO autograd
    // TODO pinned memory
  }) {
    m ??= n;
    final arena = ffi.Arena();
    try {
      final options = FFITensorOptions.make(
        dataType: dtype,
        device: device,
        layout: layout,
        allocator: arena,
      );
      final tensor = TensorFFI.eye(n, m, options.ref);
      return Tensor(tensor);
    } finally {
      arena.releaseAll();
    }
  }

  static Tensor fromBlob(
    ffi.Pointer<ffi.Void> dataPointer,
    List<int> sizes, {
    required DataType dtype,
    Device device = Device.cpu,
    Layout layout = Layout.strided,
    // TODO memory format
    // TODO autograd
    // TODO pinned memory
  }) {
    final arena = ffi.Arena();
    try {
      final options = FFITensorOptions.make(
        dataType: dtype,
        device: device,
        layout: layout,
        allocator: arena,
      );
      final sizesPointer = arena.allocate<ffi.Int64>(sizes.length);
      sizesPointer.asTypedList(sizes.length).setAll(0, sizes);
      final tensor = TensorFFI.fromBlob(
        dataPointer,
        sizesPointer,
        sizes.length,
        options.ref,
      );
      return Tensor(tensor);
    } finally {
      arena.releaseAll();
    }
  }

  int get dim => TensorFFI.dim(nativePtr);

  List<int> get sizes {
    final dim = this.dim;
    final sizesPtr = ffi.malloc.allocate<ffi.Int64>(dim);
    try {
      TensorFFI.sizes(nativePtr, dim, sizesPtr);
      return sizesPtr.asTypedList(dim).toList();
    } finally {
      ffi.malloc.free(sizesPtr);
    }
  }

  List<int> get shape => sizes;

  Device get device {
    final device = TensorFFI.tensorGetDevice(nativePtr);
    return Device(
      deviceType: DeviceType.fromId(device.deviceType),
      deviceIndex: device.deviceIndex,
    );
  }

  bool get isScalar => shape.isEmpty;

  dynamic get scalar {
    if (!isScalar) {
      throw Exception('Tensor is not a scalar');
    }
    final scalar = TensorFFI.item(nativePtr);
    return scalar.value;
  }

  Tensor operator [](int index) => get(index);

  Tensor get(int index) {
    if (isScalar) {
      throw Exception('Scalar tensor cannot be indexed');
    }
    final int max = shape[0];
    if (index >= max) {
      throw IndexError.withLength(index, max);
    }
    try {
      final tensor = TensorFFI.get(nativePtr, index);
      return Tensor(tensor);
    } catch (e) {
      print(e);
      throw Exception('Index out of bounds');
    }
  }

  Tensor index(List<dynamic> indices) {
    final arena = ffi.Arena();
    try {
      final indicesPointer = arena.allocate<FFIIndex>(
        ffi.sizeOf<FFIIndex>() * indices.length,
      );
      for (int i = 0; i < indices.length; i++) {
        final index = indices[i];
        (indicesPointer + i).ref.fromIndex(index, arena);
      }
      final tensor = TensorFFI.index(nativePtr, indicesPointer, indices.length);
      return Tensor(tensor);
    } finally {
      arena.releaseAll();
    }
  }

  Tensor view(List<int> sizes) {
    final arena = ffi.Arena();
    try {
      final sizesPointer = arena.allocate<ffi.Int64>(
        ffi.sizeOf<ffi.Int64>() * sizes.length,
      );
      sizesPointer.asTypedList(sizes.length).setAll(0, sizes);
      final tensor = TensorFFI.view(nativePtr, sizesPointer, sizes.length);
      return Tensor(tensor);
    } finally {
      arena.releaseAll();
    }
  }

  Tensor reshape(List<int> sizes) {
    throw UnimplementedError();
  }

  Tensor expand(List<int> sizes) {
    final arena = ffi.Arena();
    try {
      final sizesPointer = arena.allocate<ffi.Int64>(
        ffi.sizeOf<ffi.Int64>() * sizes.length,
      );
      sizesPointer.asTypedList(sizes.length).setAll(0, sizes);
      final tensor = TensorFFI.expand(
        nativePtr,
        sizesPointer,
        sizes.length,
        false,
      );
      return Tensor(tensor);
    } finally {
      arena.releaseAll();
    }
  }

  DataType get dataType {
    // TODO
    throw UnimplementedError();
  }

  Tensor to({DataType? dataType, Device? device, Layout? layout}) {
    // TODO
    throw UnimplementedError();
  }

  Tensor contiguous() {
    // TODO
    throw UnimplementedError();
  }

  Tensor transpose(List<int> dims) {
    // TODO
    throw UnimplementedError();
  }

  Tensor operator +(dynamic /* Tensor | num */ other) {
    final arena = ffi.Arena();
    try {
      if (other is Tensor) {
        final alpha = FFIScalar.allocate(arena);
        alpha.ref.setInt(1);
        final tensor = TensorFFI.addition(
          nativePtr,
          other.nativePtr,
          alpha.ref,
        );
        return Tensor(tensor);
      } else if (other is num) {
        throw UnimplementedError('operator+num not implemented for Tensor');
      } else if (other is (Tensor, dynamic)) {
        final alpha = FFIScalar.allocate(arena);
        alpha.ref.setValue(other.$2);
        final tensor = TensorFFI.addition(
          nativePtr,
          other.$1.nativePtr,
          alpha.ref,
        );
        return Tensor(tensor);
      } else if (other is (num, dynamic)) {
        throw UnimplementedError('operator+num not implemented for Tensor');
      }
      throw UnimplementedError(
        'operator+(${other.runtimeType}) not implemented for Tensor',
      );
    } finally {
      arena.releaseAll();
    }
  }

  Tensor operator -(dynamic /* Tensor | num */ other) {
    final arena = ffi.Arena();
    try {
      if (other is Tensor) {
        final alpha = FFIScalar.allocate(arena);
        alpha.ref.setInt(1);
        final tensor = TensorFFI.subtraction(
          nativePtr,
          other.nativePtr,
          alpha.ref,
        );
        return Tensor(tensor);
      } else if (other is num) {
        throw UnimplementedError('operator+num not implemented for Tensor');
      } else if (other is (Tensor, dynamic)) {
        final alpha = FFIScalar.allocate(arena);
        alpha.ref.setValue(other.$2);
        final tensor = TensorFFI.subtraction(
          nativePtr,
          other.$1.nativePtr,
          alpha.ref,
        );
        return Tensor(tensor);
      } else if (other is (num, dynamic)) {
        throw UnimplementedError('operator+num not implemented for Tensor');
      }
      throw UnimplementedError(
        'operator+(${other.runtimeType}) not implemented for Tensor',
      );
    } finally {
      arena.releaseAll();
    }
  }

  Tensor operator *(dynamic /* Tensor | num */ other) {
    final arena = ffi.Arena();
    try {
      if (other is Tensor) {
        final tensor = TensorFFI.multiplication(nativePtr, other.nativePtr);
        return Tensor(tensor);
      } else if (other is num) {
        throw UnimplementedError('operator+num not implemented for Tensor');
      } else if (other is (num, dynamic)) {
        throw UnimplementedError('operator+num not implemented for Tensor');
      }
      throw UnimplementedError(
        'operator+(${other.runtimeType}) not implemented for Tensor',
      );
    } finally {
      arena.releaseAll();
    }
  }

  Tensor operator /(dynamic /* Tensor | num */ other) {
    final arena = ffi.Arena();
    try {
      if (other is Tensor) {
        final tensor = TensorFFI.division(nativePtr, other.nativePtr);
        return Tensor(tensor);
      } else if (other is num) {
        throw UnimplementedError('operator/num not implemented for Tensor');
      } else if (other is (num, dynamic)) {
        throw UnimplementedError('operator/num not implemented for Tensor');
      }
      throw UnimplementedError(
        'operator/(${other.runtimeType}) not implemented for Tensor',
      );
    } finally {
      arena.releaseAll();
    }
  }

  Tensor matmul(Tensor other) {
    // TODO
    throw UnimplementedError();
  }

  Tensor softmax(int dim, {DataType? dataType}) {
    // TODO
    throw UnimplementedError();
  }

  Tensor dropout(double p, {bool training = true}) {
    // TODO
    throw UnimplementedError();
  }

  Tensor sigmoid() {
    final tensor = TensorFFI.sigmoid(nativePtr);
    return Tensor(tensor);
  }

  Tensor gelu(GeluApporimate approximate) {
    final arena = ffi.Arena();
    try {
      final activation = approximate.name.toNativeUtf8(allocator: arena);
      final tensor = TensorFFI.gelu(nativePtr, activation);
      return Tensor(tensor);
    } finally {
      arena.releaseAll();
    }
  }

  String _print1d(int size, Tensor tensor) {
    final sb = StringBuffer();
    sb.write('[');
    for (int i = 0; i < size; i++) {
      if (i > 0) sb.write(', ');
      sb.write(tensor.get(i).scalar);
      if (i == 50 && size > 100) {
        sb.write('...');
        i = size - 50;
      }
    }
    sb.write(']');
    return sb.toString();
  }

  String _print2d(int size0, int size1, Tensor tensor) {
    final sb = StringBuffer();
    sb.write('[');
    for (int i = 0; i < size0; i++) {
      if (i > 0) sb.write(',\n ');
      sb.write(_print1d(size1, tensor.get(i)));
      if (i == 50 && size0 > 100) {
        sb.write('...');
        i = size0 - 50;
      }
    }
    sb.write(']\n');
    return sb.toString();
  }

  @override
  String toString() {
    final sizes = this.sizes;
    if (sizes.isEmpty) {
      return '[$scalar]';
    } else if (sizes.length == 1) {
      return _print1d(sizes[0], this);
    } else if (sizes.length == 2) {
      return _print2d(sizes[0], sizes[1], this);
    } else {
      throw UnimplementedError();
    }
  }
}

abstract class Index {}

class NewDim implements Index {}

class Ellipsis implements Index {}

class Slice implements Index {
  final int? start;
  final int? end;
  final int step;

  Slice({this.start, this.end, this.step = 1});
}

enum GeluApporimate { none, tanh }

Tensor linear(Tensor input, Tensor weight, {Tensor? bias}) {
  final tensorPtr = TensorFFI.linear(
    input.nativePtr,
    weight.nativePtr,
    bias?.nativePtr ?? ffi.nullptr,
  );
  return Tensor(tensorPtr);
}

Tensor layerNorm(
  Tensor input,
  List<int> normalizedShape,
  Tensor? weight,
  Tensor? bias,
  double eps,
) {
  final arena = ffi.Arena();
  try {
    final normalizedShapePointer = arena.allocate<ffi.Int64>(
      ffi.sizeOf<ffi.Int64>() * normalizedShape.length,
    );
    normalizedShapePointer
        .asTypedList(normalizedShape.length)
        .setAll(0, normalizedShape);

    final tensorPtr = TensorFFI.layerNorm(
      input.nativePtr,
      normalizedShapePointer,
      normalizedShape.length,
      weight?.nativePtr ?? ffi.nullptr,
      bias?.nativePtr ?? ffi.nullptr,
      eps,
      true, // TODO: enable_cudnn
    );
    return Tensor(tensorPtr);
  } finally {
    arena.releaseAll();
  }
}

Tensor embedding(
  Tensor weights,
  Tensor indices,
  int paddingIdx,
  bool scaleGradByFreq,
  bool sparse,
) {
  final tensorPtr = TensorFFI.embedding(
    weights.nativePtr,
    indices.nativePtr,
    paddingIdx,
    scaleGradByFreq,
    sparse,
  );

  return Tensor(tensorPtr);
}

class DeviceType {
  final String name;
  final int type;

  const DeviceType(this.name, this.type);

  @override
  String toString() => name;

  static const cpu = DeviceType('CPU', 0);
  static const cuda = DeviceType('CUDA', 1);
  static const mkldnn = DeviceType('MKLDNN', 2);
  static const opengl = DeviceType('OpenGL', 3);
  static const opencl = DeviceType('OpenCL', 4);
  static const ideep = DeviceType('IDEEP', 5);
  static const hip = DeviceType('HIP', 6);
  static const fpga = DeviceType('FPGA', 7);

  /// ONNX Runtime / Microsoft
  static const maia = DeviceType('MAIA', 8);
  static const xla = DeviceType('XLA', 9);
  static const vulkan = DeviceType('Vulkan', 10);
  static const metal = DeviceType('Metal', 11);
  static const xpu = DeviceType('XPU', 12);
  static const mps = DeviceType('MPS', 13);

  /// Meta (tensors with no data)
  static const meta = DeviceType('Meta', 14);

  /// HPU / HABANA
  static const hpu = DeviceType('HPU', 15);
  // // SX-Aurora / NEC
  static const ve = DeviceType('VE', 16);

  /// Lazy Tensors
  static const lazy = DeviceType('Lazy', 17);

  /// Graphcore IPU
  static const ipu = DeviceType('IPU', 18);

  /// Meta training and inference devices
  static const mtia = DeviceType('MTIA', 19);

  static DeviceType fromId(int type) =>
      _byId[type] ?? DeviceType('Unknown', type);

  static final Map<int, DeviceType> _byId = Map.fromEntries(
    list.map((v) => MapEntry(v.type, v)),
  );

  static const List<DeviceType> list = [
    cpu,
    cuda,
    mkldnn,
    opengl,
    opencl,
    ideep,
    hip,
    fpga,
    maia,
    xla,
    vulkan,
    metal,
    xpu,
    mps,
    meta,
    hpu,
    ve,
    lazy,
    ipu,
    mtia,
  ];
}

class Device {
  final DeviceType deviceType;
  final int deviceIndex;

  const Device({required this.deviceType, required this.deviceIndex});

  @override
  String toString() => '$deviceType:$deviceIndex';

  static const cpu = Device(deviceType: DeviceType.cpu, deviceIndex: -1);
}

class DataType {
  final String? name;
  final int type;
  final String? safetensorName;
  final int numBytes;

  const DataType({
    required this.name,
    required this.type,
    required this.safetensorName,
    this.numBytes = 0,
  });

  @override
  String toString() =>
      'DataType{name: $name, type: $type, safetensorName: $safetensorName}';

  static const uint8 = DataType(name: 'Uint8', type: 0, safetensorName: 'U8');
  static const int8 = DataType(name: 'Int8', type: 1, safetensorName: 'I8');
  static const int16 = DataType(name: 'Int16', type: 2, safetensorName: 'I16');
  static const int32 = DataType(name: 'Int32', type: 3, safetensorName: 'I32');
  static const int64 = DataType(name: 'Int64', type: 4, safetensorName: 'I64');
  static const half = DataType(name: 'Half', type: 5, safetensorName: 'F16');
  static const float = DataType(name: 'Float', type: 6, safetensorName: 'F32');
  static const float64 = DataType(
    name: 'Float64',
    type: 7,
    safetensorName: 'F64',
  );
  static const complexHalf = DataType(
    name: 'ComplexHalf',
    type: 8,
    safetensorName: null,
  );
  static const complexFloat = DataType(
    name: 'ComplexFloat',
    type: 9,
    safetensorName: null,
  );
  static const complexDouble = DataType(
    name: 'ComplexDouble',
    type: 10,
    safetensorName: null,
  );
  static const boolean = DataType(
    name: 'Bool',
    type: 11,
    safetensorName: 'BOOL',
  );
  static const qInt8 = DataType(name: 'QInt8', type: 12, safetensorName: null);
  static const qUInt8 = DataType(
    name: 'QUInt8',
    type: 13,
    safetensorName: null,
  );
  static const qInt32 = DataType(
    name: 'QInt32',
    type: 14,
    safetensorName: null,
  );
  static const bFloat16 = DataType(
    name: 'BFloat16',
    type: 15,
    safetensorName: 'BF16',
  );
  static const float8e5m2 = DataType(
    name: 'Float8e5m2',
    type: 23,
    safetensorName: 'F8_E5M2',
  );
  static const float8e4m3fn = DataType(
    name: 'Float8e4m3fn',
    type: 24,
    safetensorName: null,
  );
  static const float8e5m2fnuz = DataType(
    name: 'Float8e5m2fnuz',
    type: 25,
    safetensorName: null,
  );
  static const float8e4m3fnuz = DataType(
    name: 'Float8e4m3fnuz',
    type: 26,
    safetensorName: null,
  );

  static DataType fromId(int type) =>
      _byId[type] ?? DataType(name: null, type: type, safetensorName: null);

  static final Map<int, DataType> _byId = Map.fromEntries(
    list.map((v) => MapEntry(v.type, v)),
  );

  static const List<DataType> list = [
    uint8,
    int8,
    int16,
    int32,
    int64,
    half,
    float,
    float64,
    complexHalf,
    complexFloat,
    complexDouble,
    boolean,
    qInt8,
    qUInt8,
    qInt32,
    bFloat16,
    float8e5m2,
    float8e4m3fn,
    float8e5m2fnuz,
    float8e4m3fnuz,
  ];

  static final Map<String, DataType> _bySafeTensorName = Map.fromEntries(
    list
        .where((v) => v.safetensorName != null)
        .map((v) => MapEntry(v.safetensorName!, v)),
  );

  static DataType? fromSafeTensorName(String name) => _bySafeTensorName[name];
}

class Layout {
  final String name;
  final int type;

  const Layout(this.name, this.type);

  static const strided = Layout('Strided', 0);
  static const sparse = Layout('Sparse', 1);
  static const sparseCsr = Layout('SparseCsr', 2);
  static const mkldnn = Layout('Mkldnn', 3);
  static const sparseCsc = Layout('SparseCsc', 4);
  static const sparseBsr = Layout('SparseBsr', 5);
  static const sparseBsc = Layout('SparseBsc', 6);
  static const jagged = Layout('Jagged', 7);

  static Layout fromId(int type) => _byId[type] ?? Layout('Unknown', type);

  static final Map<int, Layout> _byId = Map.fromEntries(
    list.map((v) => MapEntry(v.type, v)),
  );

  static const List<Layout> list = [
    strided,
    sparse,
    sparseCsr,
    mkldnn,
    sparseCsc,
    sparseBsr,
    sparseBsc,
    jagged,
  ];
}
