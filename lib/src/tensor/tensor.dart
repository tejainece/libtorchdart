import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart' as ffi;
import 'package:libtorchdart/src/nn/nn2d.dart';
import 'package:libtorchdart/src/tensor_ffi/tensor_ffi.dart';

class Tensor implements ffi.Finalizable {
  ffi.Pointer<ffi.Void> nativePtr;

  Tensor(this.nativePtr) {
    _finalizer.attach(this, nativePtr);
  }

  static final _finalizer = ffi.NativeFinalizer(Torch.delete);

  static Tensor zeros(
    List<int> sizes, {
    Device device = Device.cpu,
    DataType dtype = DataType.float,
    Layout layout = Layout.strided,
    MemoryFormat memoryFormat = MemoryFormat.contiguous,
    // TODO autograd
    // TODO pinned memory
  }) {
    final arena = ffi.Arena();
    try {
      final options = FFITensorOptions.make(
        dataType: dtype,
        device: device,
        layout: layout,
        memoryFormat: memoryFormat,
        allocator: arena,
      );
      final sizesPointer = arena.allocate<ffi.Int64>(
        ffi.sizeOf<ffi.Int64>() * sizes.length,
      );
      sizesPointer.asTypedList(sizes.length).setAll(0, sizes);
      final tensor = Torch.zeros(sizesPointer, sizes.length, options.ref);
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
    MemoryFormat memoryFormat = MemoryFormat.contiguous,
    // TODO autograd
    // TODO pinned memory
  }) {
    final arena = ffi.Arena();
    try {
      final options = FFITensorOptions.make(
        dataType: dtype,
        device: device,
        layout: layout,
        memoryFormat: memoryFormat,
        allocator: arena,
      );
      final sizesPointer = arena.allocate<ffi.Int64>(
        ffi.sizeOf<ffi.Int64>() * sizes.length,
      );
      sizesPointer.asTypedList(sizes.length).setAll(0, sizes);
      final tensor = Torch.ones(sizesPointer, sizes.length, options.ref);
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
    MemoryFormat memoryFormat = MemoryFormat.contiguous,
    // TODO autograd
    // TODO pinned memory
  }) {
    final arena = ffi.Arena();
    try {
      final options = FFITensorOptions.make(
        dataType: dtype,
        device: device,
        layout: layout,
        memoryFormat: memoryFormat,
        allocator: arena,
      );
      final tensor = Torch.arange(end, options.ref);
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
    MemoryFormat memoryFormat = MemoryFormat.contiguous,
    // TODO autograd
    // TODO pinned memory
  }) {
    final arena = ffi.Arena();
    try {
      final options = FFITensorOptions.make(
        dataType: dtype,
        device: device,
        layout: layout,
        memoryFormat: memoryFormat,
        allocator: arena,
      );
      final sizesPointer = arena.allocate<ffi.Int64>(
        ffi.sizeOf<ffi.Int64>() * sizes.length,
      );
      sizesPointer.asTypedList(sizes.length).setAll(0, sizes);
      final tensor = Torch.rand(sizesPointer, sizes.length, options.ref);
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
    MemoryFormat memoryFormat = MemoryFormat.contiguous,
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
        memoryFormat: memoryFormat,
        allocator: arena,
      );
      final tensor = Torch.eye(n, m, options.ref);
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
    MemoryFormat memoryFormat = MemoryFormat.contiguous,
    // TODO autograd
    // TODO pinned memory
  }) {
    final arena = ffi.Arena();
    try {
      final options = FFITensorOptions.make(
        dataType: dtype,
        device: device,
        layout: layout,
        memoryFormat: memoryFormat,
        allocator: arena,
      );
      final sizesPointer = arena.allocate<ffi.Int64>(sizes.length);
      sizesPointer.asTypedList(sizes.length).setAll(0, sizes);
      final tensor = Torch.fromBlob(
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

  int get dim => Torch.dim(nativePtr);

  List<int> get sizes {
    final dim = this.dim;
    final arena = ffi.Arena();
    try {
      final sizesPtr = arena.allocate<ffi.Int64>(ffi.sizeOf<ffi.Int64>() * dim);
      Torch.sizes(nativePtr, dim, sizesPtr);
      return sizesPtr.asTypedList(dim).toList();
    } finally {
      arena.releaseAll();
    }
  }

  List<int> get shape => sizes;

  Device get device {
    final device = Torch.tensorGetDevice(nativePtr);
    return Device(
      deviceType: DeviceType.fromId(device.deviceType),
      deviceIndex: device.deviceIndex,
    );
  }

  bool get isScalar => shape.isEmpty;

  dynamic get scalar {
    final scalar = Torch.scalar(nativePtr);
    return scalar.value;
  }

  dynamic scalarAt(int index) {
    final scalar = Torch.scalarAt(nativePtr, index);
    return scalar.value;
  }

  Tensor operator [](int index) => get(index);

  Tensor get(int index) {
    /*if (isScalar) {
      throw Exception('Scalar tensor cannot be indexed');
    }
    final int max = shape[0];
    if (index >= max) {
      throw IndexError.withLength(index, max);
    }*/
    try {
      final tensor = Torch.get(nativePtr, index);
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
      final tensor = Torch.index(nativePtr, indicesPointer, indices.length);
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
      final tensor = Torch.view(nativePtr, sizesPointer, sizes.length);
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
      final tensor = Torch.expand(nativePtr, sizesPointer, sizes.length, false);
      return Tensor(tensor);
    } finally {
      arena.releaseAll();
    }
  }

  Tensor permute(List<int> dims) {
    final arena = ffi.Arena();
    try {
      final dimsPointer = arena.allocate<ffi.Int64>(
        ffi.sizeOf<ffi.Int64>() * dims.length,
      );
      dimsPointer.asTypedList(dims.length).setAll(0, dims);
      final tensor = Torch.permute(nativePtr, dimsPointer, dims.length);
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

  Tensor contiguous({MemoryFormat format = MemoryFormat.contiguous}) {
    final tensor = Torch.contiguous(nativePtr, format.id);
    return Tensor(tensor);
  }

  /// Returns the transposed version of the tensor. Swaps [dim0] and [dim1].
  Tensor transpose(int dim0, int dim1) {
    final tensor = Torch.transpose(nativePtr, dim0, dim1);
    return Tensor(tensor);
  }

  Tensor pad(List<int> pad, {PadMode mode = PadMode.constant, double? value}) {
    final arena = ffi.Arena();
    try {
      final padPointer = arena.allocate<ffi.Int64>(
        ffi.sizeOf<ffi.Int64>() * pad.length,
      );
      padPointer.asTypedList(pad.length).setAll(0, pad);

      ffi.Pointer<ffi.Double> valuePointer = ffi.nullptr;
      if (value != null) {
        valuePointer = arena.allocate<ffi.Double>(ffi.sizeOf<ffi.Double>());
        valuePointer.value = value;
      }

      final tensor = Torch.pad(
        nativePtr,
        padPointer,
        pad.length,
        mode.index,
        valuePointer,
      );
      return Tensor(tensor);
    } finally {
      arena.releaseAll();
    }
  }

  Tensor operator +(dynamic /* Tensor | num */ other) {
    final arena = ffi.Arena();
    try {
      if (other is Tensor) {
        final alpha = FFIScalar.allocate(arena);
        alpha.ref.setInt(1);
        final tensor = Torch.addition(nativePtr, other.nativePtr, alpha.ref);
        return Tensor(tensor);
      } else if (other is num) {
        throw UnimplementedError('operator+num not implemented for Tensor');
      } else if (other is (Tensor, dynamic)) {
        final alpha = FFIScalar.allocate(arena);
        alpha.ref.setValue(other.$2);
        final tensor = Torch.addition(nativePtr, other.$1.nativePtr, alpha.ref);
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
        final tensor = Torch.subtraction(nativePtr, other.nativePtr, alpha.ref);
        return Tensor(tensor);
      } else if (other is num) {
        throw UnimplementedError('operator+num not implemented for Tensor');
      } else if (other is (Tensor, dynamic)) {
        final alpha = FFIScalar.allocate(arena);
        alpha.ref.setValue(other.$2);
        final tensor = Torch.subtraction(
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
        final tensor = Torch.multiplication(nativePtr, other.nativePtr);
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
        final tensor = Torch.division(nativePtr, other.nativePtr);
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
    final tensor = Torch.matmul(nativePtr, other.nativePtr);
    return Tensor(tensor);
  }

  Tensor softmax(int dim, {DataType? dataType}) {
    final arena = ffi.Arena();
    try {
      ffi.Pointer<ffi.Int8> dataTypePointer = ffi.nullptr;
      if (dataType != null) {
        dataTypePointer = arena.allocate<ffi.Int8>(ffi.sizeOf<ffi.Int8>());
        dataTypePointer.value = dataType.type;
      }
      final tensor = Torch.softmax(nativePtr, dim, dataTypePointer);
      return Tensor(tensor);
    } finally {
      arena.releaseAll();
    }
  }

  Tensor layerNorm(
    List<int> normalizedShape, {
    Tensor? weight,
    Tensor? bias,
    double eps = 1e-5,
  }) {
    final arena = ffi.Arena();
    try {
      final normalizedShapePointer = arena.allocate<ffi.Int64>(
        ffi.sizeOf<ffi.Int64>() * normalizedShape.length,
      );
      normalizedShapePointer
          .asTypedList(normalizedShape.length)
          .setAll(0, normalizedShape);

      final tensorPtr = Torch.layerNorm(
        nativePtr,
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

  Tensor groupNorm(
    int numGroups, {
    Tensor? weight,
    Tensor? bias,
    double eps = 1e-5,
  }) {
    final tensorPtr = Torch.groupNorm(
      nativePtr,
      numGroups,
      weight?.nativePtr ?? ffi.nullptr,
      bias?.nativePtr ?? ffi.nullptr,
      eps,
    );
    return Tensor(tensorPtr);
  }

  Tensor dropout(double p, {bool training = true}) {
    final tensor = Torch.dropout(nativePtr, p, training);
    return Tensor(tensor);
  }

  Tensor sigmoid() {
    final tensor = Torch.sigmoid(nativePtr);
    return Tensor(tensor);
  }

  Tensor relu() {
    final tensor = Torch.relu(nativePtr);
    return Tensor(tensor);
  }

  Tensor gelu(GeluApporimate approximate) {
    final arena = ffi.Arena();
    try {
      final activation = approximate.name.toNativeUtf8(allocator: arena);
      final tensor = Torch.gelu(nativePtr, activation);
      return Tensor(tensor);
    } finally {
      arena.releaseAll();
    }
  }

  Tensor silu() {
    final tensor = Torch.silu(nativePtr);
    return Tensor(tensor);
  }

  static void _print1d(StringBuffer sb, int size, Tensor tensor) {
    sb.write('[');
    for (int i = 0; i < size; i++) {
      if (i > 0) sb.write(', ');
      sb.write(tensor.scalarAt(i));
      if (i == 50 && size > 100) {
        sb.write(', ..............');
        i = size - 50;
      }
    }
    sb.write(']');
  }

  static void _print2d(StringBuffer sb, int size0, int size1, Tensor tensor) {
    sb.write('[');
    for (int i = 0; i < size0; i++) {
      _print1d(sb, size1, tensor.get(i));
      if (i == 50 && size0 > 100) {
        sb.writeln(',\n ..............');
        i = size0 - 50;
      } else {
        if (i != size0 - 1) sb.write(',\n ');
      }
    }
    sb.write(']');
  }

  static void _printDim(
    StringBuffer sb,
    List<int> indexPrefix,
    List<int> sizes,
    Tensor tensor,
  ) {
    final dim = sizes.length;
    if (dim < 3) {
      throw Exception('_printDim called with tensor dim less than 3');
    }

    final count = tensor.shape[0];
    if (dim == 3) {
      for (int i = 0; i < count; i++) {
        sb.writeln('(${(indexPrefix.followedBy([i])).join(',')},*,*) = ');
        _print2d(sb, sizes[1], sizes[2], tensor.get(i));
        if (i < count - 1) {
          sb.writeln();
        }
      }
    } else {
      for (int i = 0; i < count; i++) {
        _printDim(
          sb,
          indexPrefix.followedBy([i]).toList(),
          sizes.sublist(1),
          tensor.get(i),
        );
        sb.writeln();
      }
    }
  }

  @override
  String toString() {
    final sizes = this.sizes;
    final sb = StringBuffer();
    sb.writeln('Tensor{${sizes.join(',')}}');
    if (sizes.isEmpty) {
      sb.write('[$scalar]');
    } else if (sizes.length == 1) {
      _print1d(sb, sizes[0], this);
    } else if (sizes.length == 2) {
      _print2d(sb, sizes[0], sizes[1], this);
    } else {
      _printDim(sb, [], sizes, this);
    }
    return sb.toString();
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
  final tensorPtr = Torch.linear(
    input.nativePtr,
    weight.nativePtr,
    bias?.nativePtr ?? ffi.nullptr,
  );
  return Tensor(tensorPtr);
}

Tensor embedding(
  Tensor weights,
  Tensor indices,
  int paddingIdx,
  bool scaleGradByFreq,
  bool sparse,
) {
  final tensorPtr = Torch.embedding(
    weights.nativePtr,
    indices.nativePtr,
    paddingIdx,
    scaleGradByFreq,
    sparse,
  );

  return Tensor(tensorPtr);
}

Tensor conv2d(
  Tensor input,
  Tensor weight, {
  Tensor? bias,
  SymmetricPadding2D stride = const SymmetricPadding2D(
    vertical: 1,
    horizontal: 1,
  ),
  SymmetricPadding2D padding = const SymmetricPadding2D(
    vertical: 0,
    horizontal: 0,
  ),
  SymmetricPadding2D dilation = const SymmetricPadding2D(
    vertical: 1,
    horizontal: 1,
  ),
  int groups = 1,
}) {
  final arena = ffi.Arena();
  try {
    ffi.Pointer<ffi.Int64> stridePointer = arena.allocate<ffi.Int64>(
      ffi.sizeOf<ffi.Int64>() * 2,
    );
    stridePointer.value = stride.vertical;
    (stridePointer + 1).value = stride.horizontal;

    ffi.Pointer<ffi.Int64> paddingPointer = arena.allocate<ffi.Int64>(
      ffi.sizeOf<ffi.Int64>() * 2,
    );
    paddingPointer.value = padding.vertical;
    (paddingPointer + 1).value = padding.horizontal;

    ffi.Pointer<ffi.Int64> dilationPointer = arena.allocate<ffi.Int64>(
      ffi.sizeOf<ffi.Int64>() * 2,
    );
    dilationPointer.value = dilation.vertical;
    (dilationPointer + 1).value = dilation.horizontal;

    final tensorPtr = Torch.conv2d(
      input.nativePtr,
      weight.nativePtr,
      bias?.nativePtr ?? ffi.nullptr,
      stridePointer,
      paddingPointer,
      dilationPointer,
      groups,
    );
    return Tensor(tensorPtr);
  } finally {
    arena.releaseAll();
  }
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

class MemoryFormat {
  final String name;
  final int id;
  const MemoryFormat(this.name, this.id);

  static const contiguous = MemoryFormat('Contiguous', 0);
  static const preserve = MemoryFormat('Preserve', 1);
  static const channelsLast = MemoryFormat('ChannelsLast', 2);
  static const channelsLast3d = MemoryFormat('ChannelsLast3d', 3);
}
