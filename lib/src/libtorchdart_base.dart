import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart' as ffi;
import 'package:libtorchdart/src/tensor_ffi/tensor_ffi.dart';

extension type Tensor(ffi.Pointer<ffi.Void> _tensor) {
  int get dim => TensorFFI.dim(_tensor);

  List<int> get sizes {
    final dim = this.dim;
    final sizesPtr = ffi.malloc.allocate<ffi.Int64>(dim);
    try {
      TensorFFI.sizes(_tensor, dim, sizesPtr);
      return sizesPtr.asTypedList(dim).toList();
    } finally {
      ffi.malloc.free(sizesPtr);
    }
  }

  List<int> get shape => sizes;

  Device get device {
    final device = TensorFFI.tensorGetDevice(_tensor);
    return Device(
      deviceType: DeviceType.fromId(device.deviceType),
      deviceIndex: device.deviceIndex,
    );
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
    final options = FFITensorOptions.make(
      dataType: dtype,
      device: device,
      layout: layout,
    );
    try {
      final tensor = TensorFFI.eye(n, m, options.ref);
      return Tensor(tensor);
    } finally {
      ffi.malloc.free(options);
    }
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
  final String name;
  final int type;

  const DataType(this.name, this.type);

  static const uint8 = DataType('Uint8', 0);
  static const int8 = DataType('Int8', 1);
  static const int16 = DataType('Int16', 2);
  static const int32 = DataType('Int32', 3);
  static const int64 = DataType('Int64', 4);
  static const half = DataType('Half', 5);
  static const float = DataType('Float', 6);
  static const float64 = DataType('Float64', 7);
  static const complexHalf = DataType('ComplexHalf', 8);
  static const complexFloat = DataType('ComplexFloat', 9);
  static const complexDouble = DataType('ComplexDouble', 10);
  static const boolean = DataType('Bool', 11);
  static const qInt8 = DataType('QInt8', 12);
  static const qUInt8 = DataType('QUInt8', 13);
  static const qInt32 = DataType('QInt32', 14);
  static const bFloat16 = DataType('BFloat16', 15);
  static const float8e5m2 = DataType('Float8e5m2', 23);
  static const float8e4m3fn = DataType('Float8e4m3fn', 24);
  static const float8e5m2fnuz = DataType('Float8e5m2fnuz', 25);
  static const float8e4m3fnuz = DataType('Float8e4m3fnuz', 26);

  static DataType fromId(int type) => _byId[type] ?? DataType('Unknown', type);

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
