import 'dart:ffi' as ffi;
import 'package:collection/collection.dart';
import 'package:ffi/ffi.dart' as ffi;
import 'package:tensor/tensor.dart';
import 'package:tensor/src/ffi/torch_ffi.dart';

export 'nn.dart';

class Tensor implements ffi.Finalizable {
  ffi.Pointer<ffi.Void> nativePtr;
  final bool shouldDelete;

  String? name;

  Tensor(this.nativePtr, {this.shouldDelete = true, this.name}) {
    if (shouldDelete) {
      _finalizer.attach(this, nativePtr, detach: this);
    }
  }

  static final _finalizer = ffi.NativeFinalizer(FFITensor.delete);

  void release() {
    if (shouldDelete) {
      _finalizer.detach(this);
      FFITensor.deleteTensor(nativePtr);
    }
  }

  static Tensor empty(
    List<int> sizes, {
    String? name,
    Device? device,
    DataType? datatype,
    Layout? layout,
    MemoryFormat? memoryFormat,
    bool? requiresGrad,
    bool? pinnedMemory,
  }) {
    final arena = ffi.Arena();
    try {
      final options = CTensorOptions.make(
        dataType: datatype,
        device: device,
        layout: layout,
        memoryFormat: memoryFormat,
        requiresGrad: requiresGrad,
        pinnedMemory: pinnedMemory,
        allocator: arena,
      );
      final sizesPointer = arena.allocate<ffi.Int64>(
        ffi.sizeOf<ffi.Int64>() * sizes.length,
      );
      sizesPointer.asTypedList(sizes.length).setAll(0, sizes);
      final tensor = FFITensor.empty(sizesPointer, sizes.length, options.ref);
      return Tensor(tensor, name: name);
    } finally {
      arena.releaseAll();
    }
  }

  static Tensor zeros(
    List<int> sizes, {
    String? name,
    Device? device,
    DataType? datatype,
    Layout? layout,
    MemoryFormat? memoryFormat,
    bool? requiresGrad,
    bool? pinnedMemory,
  }) {
    final arena = ffi.Arena();
    try {
      final options = CTensorOptions.make(
        dataType: datatype,
        device: device,
        layout: layout,
        memoryFormat: memoryFormat,
        requiresGrad: requiresGrad,
        pinnedMemory: pinnedMemory,
        allocator: arena,
      );
      final sizesPointer = arena.allocate<ffi.Int64>(
        ffi.sizeOf<ffi.Int64>() * sizes.length,
      );
      sizesPointer.asTypedList(sizes.length).setAll(0, sizes);
      final tensor = FFITensor.zeros(sizesPointer, sizes.length, options.ref);
      return Tensor(tensor, name: name);
    } finally {
      arena.releaseAll();
    }
  }

  static Tensor ones(
    List<int> sizes, {
    String? name,
    Device? device,
    DataType? datatype,
    Layout? layout,
    MemoryFormat? memoryFormat,
    bool? requiresGrad,
    bool? pinnedMemory,
  }) {
    final arena = ffi.Arena();
    try {
      final options = CTensorOptions.make(
        dataType: datatype,
        device: device,
        layout: layout,
        memoryFormat: memoryFormat,
        requiresGrad: requiresGrad,
        pinnedMemory: pinnedMemory,
        allocator: arena,
      );
      final sizesPointer = arena.allocate<ffi.Int64>(
        ffi.sizeOf<ffi.Int64>() * sizes.length,
      );
      sizesPointer.asTypedList(sizes.length).setAll(0, sizes);
      final tensor = FFITensor.ones(sizesPointer, sizes.length, options.ref);
      return Tensor(tensor, name: name);
    } finally {
      arena.releaseAll();
    }
  }

  static Tensor arange(
    num start,
    num end, {
    num step = 1,
    String? name,
    Device? device,
    DataType? datatype,
    Layout? layout,
    MemoryFormat? memoryFormat,
    bool? requiresGrad,
    bool? pinnedMemory,
  }) {
    final arena = ffi.Arena();
    try {
      final options = CTensorOptions.make(
        dataType: datatype,
        device: device,
        layout: layout,
        memoryFormat: memoryFormat,
        requiresGrad: requiresGrad,
        pinnedMemory: pinnedMemory,
        allocator: arena,
      );

      final tensor = FFITensor.arange(
        CScalar.allocateWithValue(arena, start),
        CScalar.allocateWithValue(arena, end),
        CScalar.allocateWithValue(arena, step),
        options.ref,
      );
      return Tensor(tensor, name: name);
    } finally {
      arena.releaseAll();
    }
  }

  static Tensor rand(
    List<int> sizes, {
    String? name,
    Generator? generator,
    Device? device,
    DataType? datatype,
    Layout? layout,
    MemoryFormat? memoryFormat,
    bool? requiresGrad,
    bool? pinnedMemory,
  }) {
    CGenerator cGenerator = generator?.nativePtr ?? ffi.nullptr;
    final arena = ffi.Arena();
    try {
      final options = CTensorOptions.make(
        dataType: datatype,
        device: device,
        layout: layout,
        memoryFormat: memoryFormat,
        requiresGrad: requiresGrad,
        pinnedMemory: pinnedMemory,
        allocator: arena,
      );
      final sizesPointer = arena.allocate<ffi.Int64>(
        ffi.sizeOf<ffi.Int64>() * sizes.length,
      );
      sizesPointer.asTypedList(sizes.length).setAll(0, sizes);
      final tensor = FFITensor.rand(
        sizesPointer,
        sizes.length,
        cGenerator,
        options.ref,
      );
      return Tensor(tensor, name: name);
    } finally {
      arena.releaseAll();
    }
  }

  static Tensor randn(
    List<int> sizes, {
    String? name,
    Generator? generator,
    Device? device,
    DataType? datatype,
    Layout? layout,
    MemoryFormat? memoryFormat,
    bool? requiresGrad,
    bool? pinnedMemory,
  }) {
    CGenerator cGenerator = generator?.nativePtr ?? ffi.nullptr;
    final arena = ffi.Arena();
    try {
      final options = CTensorOptions.make(
        dataType: datatype,
        device: device,
        layout: layout,
        memoryFormat: memoryFormat,
        requiresGrad: requiresGrad,
        pinnedMemory: pinnedMemory,
        allocator: arena,
      );
      final sizesPointer = arena.allocate<ffi.Int64>(
        ffi.sizeOf<ffi.Int64>() * sizes.length,
      );
      sizesPointer.asTypedList(sizes.length).setAll(0, sizes);
      final tensor = FFITensor.randn(
        sizesPointer,
        sizes.length,
        cGenerator,
        options.ref,
      );
      return Tensor(tensor, name: name);
    } finally {
      arena.releaseAll();
    }
  }

  static Tensor randint(
    int high,
    List<int> sizes, {
    int low = 0,
    String? name,
    Generator? generator,
    Device? device,
    DataType? datatype,
    Layout? layout,
    MemoryFormat? memoryFormat,
    bool? requiresGrad,
    bool? pinnedMemory,
  }) {
    CGenerator cGenerator = generator?.nativePtr ?? ffi.nullptr;
    final arena = ffi.Arena();
    try {
      final options = CTensorOptions.make(
        dataType: datatype,
        device: device,
        layout: layout,
        memoryFormat: memoryFormat,
        requiresGrad: requiresGrad,
        pinnedMemory: pinnedMemory,
        allocator: arena,
      );
      final sizesPointer = arena.allocate<ffi.Int64>(
        ffi.sizeOf<ffi.Int64>() * sizes.length,
      );
      sizesPointer.asTypedList(sizes.length).setAll(0, sizes);
      final tensor = FFITensor.randint(
        low,
        high,
        sizesPointer,
        sizes.length,
        cGenerator,
        options.ref,
      );
      return Tensor(tensor, name: name);
    } finally {
      arena.releaseAll();
    }
  }

  static Tensor eye(
    int n, {
    String? name,
    int? m,
    Device? device,
    DataType? datatype,
    Layout? layout,
    MemoryFormat? memoryFormat,
    bool? requiresGrad,
    bool? pinnedMemory,
  }) {
    m ??= n;
    final arena = ffi.Arena();
    try {
      final options = CTensorOptions.make(
        dataType: datatype,
        device: device,
        layout: layout,
        memoryFormat: memoryFormat,
        requiresGrad: requiresGrad,
        pinnedMemory: pinnedMemory,
        allocator: arena,
      );
      final tensor = FFITensor.eye(n, m, options.ref);
      return Tensor(tensor, name: name);
    } finally {
      arena.releaseAll();
    }
  }

  static Tensor from(
    // TODO this should also accept int data and TypedList
    List<num> data,
    List<int> sizes, {
    String? name,
    required DataType datatype,
    Device? device,
    Layout? layout,
    MemoryFormat? memoryFormat,
    bool? requiresGrad,
    bool? pinnedMemory,
  }) {
    final arena = ffi.Arena();
    try {
      ffi.Pointer<ffi.Void> dataPointer;
      if (datatype == DataType.float32) {
        final ptr = arena.allocate<ffi.Float>(
          data.length * ffi.sizeOf<ffi.Float>(),
        );
        ptr.asTypedList(data.length).setAll(0, data.cast<double>());
        dataPointer = ptr.cast<ffi.Void>();
      } else if (datatype == DataType.float64) {
        final ptr = arena.allocate<ffi.Double>(
          data.length * ffi.sizeOf<ffi.Double>(),
        );
        ptr.asTypedList(data.length).setAll(0, data.cast<double>());
        dataPointer = ptr.cast<ffi.Void>();
      } else if (datatype == DataType.int64) {
        final ptr = arena.allocate<ffi.Int64>(
          data.length * ffi.sizeOf<ffi.Int64>(),
        );
        ptr.asTypedList(data.length).setAll(0, data.cast<int>());
        dataPointer = ptr.cast<ffi.Void>();
      } else if (datatype == DataType.int32) {
        final ptr = arena.allocate<ffi.Int32>(
          data.length * ffi.sizeOf<ffi.Int32>(),
        );
        ptr.asTypedList(data.length).setAll(0, data.cast<int>());
        dataPointer = ptr.cast<ffi.Void>();
      } else if (datatype == DataType.int16) {
        final ptr = arena.allocate<ffi.Int16>(
          data.length * ffi.sizeOf<ffi.Int16>(),
        );
        ptr.asTypedList(data.length).setAll(0, data.cast<int>());
        dataPointer = ptr.cast<ffi.Void>();
      } else if (datatype == DataType.int8) {
        final ptr = arena.allocate<ffi.Int8>(
          data.length * ffi.sizeOf<ffi.Int8>(),
        );
        ptr.asTypedList(data.length).setAll(0, data.cast<int>());
        dataPointer = ptr.cast<ffi.Void>();
      } else {
        throw Exception('Unsupported data type: $datatype');
      }

      final options = CTensorOptions.make(
        dataType: datatype,
        device: device,
        layout: layout,
        memoryFormat: memoryFormat,
        requiresGrad: requiresGrad,
        pinnedMemory: pinnedMemory,
        allocator: arena,
      );

      final sizesPointer = arena.allocate<ffi.Int64>(
        sizes.length * ffi.sizeOf<ffi.Int64>(),
      );
      sizesPointer.asTypedList(sizes.length).setAll(0, sizes);
      // TODO check if there is way to avoid clone
      final tensor = FFITensor.fromBlob(
        dataPointer.cast<ffi.Void>(),
        sizesPointer,
        sizes.length,
        options.ref,
      );
      return Tensor(tensor, name: name).clone();
    } finally {
      arena.releaseAll();
    }
  }

  static Tensor fromBlob(
    ffi.Pointer<ffi.Void> dataPointer,
    List<int> sizes, {
    String? name,
    required DataType datatype,
    Device? device,
    Layout? layout,
    MemoryFormat? memoryFormat,
    bool? requiresGrad,
    bool? pinnedMemory,
  }) {
    final arena = ffi.Arena();
    try {
      final options = CTensorOptions.make(
        dataType: datatype,
        device: device,
        layout: layout,
        memoryFormat: memoryFormat,
        requiresGrad: requiresGrad,
        pinnedMemory: pinnedMemory,
        allocator: arena,
      );

      final sizesPointer = arena.allocate<ffi.Int64>(
        sizes.length * ffi.sizeOf<ffi.Int64>(),
      );
      sizesPointer.asTypedList(sizes.length).setAll(0, sizes);
      final tensor = FFITensor.fromBlob(
        dataPointer,
        sizesPointer,
        sizes.length,
        options.ref,
      );
      return Tensor(tensor, name: name);
    } finally {
      arena.releaseAll();
    }
  }

  void ones_() {
    FFITensor.ones_(nativePtr);
  }

  void zeros_() {
    FFITensor.zeros_(nativePtr);
  }

  void eye_() {
    FFITensor.eye_(nativePtr);
  }

  void fill_(dynamic value) {
    final arena = ffi.Arena();
    try {
      final scalar = CScalar.allocate(arena);
      scalar.ref.setValue(value);
      FFITensor.fill_(nativePtr, scalar.ref);
    } finally {
      arena.releaseAll();
    }
  }

  void rand_({Generator? generator}) {
    CGenerator cGenerator = generator?.nativePtr ?? ffi.nullptr;
    FFITensor.rand_(nativePtr, cGenerator);
  }

  void normal_({double mean = 0.0, double std = 1.0, Generator? generator}) {
    CGenerator cGenerator = generator?.nativePtr ?? ffi.nullptr;
    FFITensor.normal_(nativePtr, cGenerator, mean, std);
  }

  void uniform_({double from = 0.0, double to = 1.0, Generator? generator}) {
    CGenerator cGenerator = generator?.nativePtr ?? ffi.nullptr;
    FFITensor.uniform_(nativePtr, cGenerator, from, to);
  }

  ffi.Pointer<void> get dataPointer => FFITensor.dataPointer(nativePtr);

  int get dim => FFITensor.dim(nativePtr);

  int get elementSize => FFITensor.elementSize(nativePtr);

  int get memorySize => elementSize * numel;

  List<int> get sizes {
    final dim = this.dim;
    final arena = ffi.Arena();
    try {
      final sizesPtr = arena.allocate<ffi.Int64>(ffi.sizeOf<ffi.Int64>() * dim);
      FFITensor.sizes(nativePtr, dim, sizesPtr);
      return sizesPtr.asTypedList(dim).toList();
    } finally {
      arena.releaseAll();
    }
  }

  List<int> get shape => sizes;

  // TODO make this native call
  int get numel => shape.reduce((a, b) => a * b);

  Device get device {
    final device = FFITensor.tensorGetDevice(nativePtr);
    return Device(
      deviceType: DeviceType.fromId(device.deviceType),
      deviceIndex: device.deviceIndex,
    );
  }

  bool get isScalar => shape.isEmpty;

  dynamic get scalar {
    final scalar = FFITensor.scalar(nativePtr);
    return scalar.value;
  }

  dynamic scalarAt(int index) {
    final errorPtr = ffi.malloc.allocate<ffi.Pointer<ffi.Utf8>>(
      ffi.sizeOf<ffi.Pointer<ffi.Utf8>>(),
    );
    try {
      errorPtr.value = ffi.nullptr;
      final scalar = FFITensor.scalarAt(nativePtr, index, errorPtr);
      if (errorPtr.value != ffi.nullptr) {
        final error = errorPtr.value.toDartString();
        throw Exception(error);
      }
      return scalar.value;
    } finally {
      final dataPtr = errorPtr.value;
      if (dataPtr != ffi.nullptr) ffi.malloc.free(dataPtr);
      ffi.malloc.free(errorPtr);
    }
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
      final tensor = FFITensor.get(nativePtr, index);
      return Tensor(tensor);
    } catch (e) {
      print(e);
      throw Exception('Index out of bounds');
    }
  }

  // TODO improve and dartify the arguments
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
      final tensor = FFITensor.index(nativePtr, indicesPointer, indices.length);
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
      final tensor = FFITensor.view(nativePtr, sizesPointer, sizes.length);
      return Tensor(tensor);
    } finally {
      arena.releaseAll();
    }
  }

  List<Tensor> splitEqually(int splitSize, {int dim = 0}) {
    final tensorPtrs = FFITensor.splitEqually(nativePtr, splitSize, dim);
    try {
      final List<Tensor> tensors = [];
      ffi.Pointer<CTensor> index = tensorPtrs;
      while (index.value != ffi.nullptr) {
        tensors.add(Tensor(index.value));
        index = index + 1;
      }
      return tensors;
    } finally {
      ffi.malloc.free(tensorPtrs);
    }
  }

  List<Tensor> split(List<int> splitSizes, {int dim = 0}) {
    final arena = ffi.Arena();
    try {
      final splitSizesPointer = arena.allocate<ffi.Int64>(
        ffi.sizeOf<ffi.Int64>() * splitSizes.length,
      );
      splitSizesPointer.asTypedList(splitSizes.length).setAll(0, splitSizes);
      final tensorPtrs = FFITensor.split(
        nativePtr,
        splitSizesPointer,
        splitSizes.length,
        dim,
      );
      final List<Tensor> tensors = [];
      ffi.Pointer<CTensor> index = tensorPtrs;
      for (int i = 0; i < splitSizes.length; i++) {
        tensors.add(Tensor(index.value));
        index = index + 1;
      }
      return tensors;
    } finally {
      arena.releaseAll();
    }
  }

  List<Tensor> chunk(int chunks, {int dim = 0}) {
    final tensorPtrs = FFITensor.chunk(nativePtr, chunks, dim);
    try {
      final List<Tensor> tensors = [];
      ffi.Pointer<CTensor> index = tensorPtrs;
      while (index.value != ffi.nullptr) {
        tensors.add(Tensor(index.value));
        index = index + 1;
      }
      return tensors;
    } finally {
      ffi.malloc.free(tensorPtrs);
    }
  }

  Tensor reshape(List<int> sizes) {
    final arena = ffi.Arena();
    try {
      final sizesPointer = arena.allocate<ffi.Int64>(
        ffi.sizeOf<ffi.Int64>() * sizes.length,
      );
      sizesPointer.asTypedList(sizes.length).setAll(0, sizes);
      final tensor = FFITensor.reshape(nativePtr, sizesPointer, sizes.length);
      return Tensor(tensor);
    } finally {
      arena.releaseAll();
    }
  }

  /// Flattens input by reshaping it into a one-dimensional tensor. If start_dim or end_dim
  /// are passed, only dimensions starting with start_dim and ending with end_dim are
  /// flattened. The order of elements in input is unchanged.
  ///
  /// This function may return the original object, a view, or copy. If no dimensions
  /// are flattened, then the original object input is returned. Otherwise, if input
  /// can be viewed as the flattened shape, then that view is returned. Finally,
  /// only if the input cannot be viewed as the flattened shape is input’s data copied.
  Tensor flatten({int startDim = 0, int endDim = -1}) {
    final tensor = FFITensor.flatten(nativePtr, startDim, endDim);
    return Tensor(tensor);
  }

  /// Returns a new view of [this] tensor with the singleton dimensions expanded to a given larger size.
  ///
  /// Passing -1 as the size for a dimension means not changing the size of the dimensions.
  ///
  /// Tensor can be also expanded to a larger number of dimensions, and the new ones will
  /// be appended at the front. For the new dimensions, the size cannot be set to -1.
  ///
  /// Expanding a tensor does not allocate new memory, but only creates a new view on the
  /// existing tensor where a dimension of size one is expanded to a larger size by
  /// setting the stride to 0. Any dimension of size 1 can be expanded to an
  /// arbitrary value without allocating new memory.
  ///
  /// To expand non-singleton dimensions to a larger size, use [repeat]. Note that [repeat]
  /// always copies data while [expand] does not.
  Tensor expand(List<int> sizes) {
    final arena = ffi.Arena();
    try {
      final sizesPointer = arena.allocate<ffi.Int64>(
        ffi.sizeOf<ffi.Int64>() * sizes.length,
      );
      sizesPointer.asTypedList(sizes.length).setAll(0, sizes);
      final tensor = FFITensor.expand(
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

  Tensor repeat(List<int> sizes) {
    final arena = ffi.Arena();
    try {
      final sizesPointer = arena.allocate<ffi.Int64>(
        ffi.sizeOf<ffi.Int64>() * sizes.length,
      );
      sizesPointer.asTypedList(sizes.length).setAll(0, sizes);
      final tensor = FFITensor.repeat(nativePtr, sizesPointer, sizes.length);
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
      final tensor = FFITensor.permute(nativePtr, dimsPointer, dims.length);
      return Tensor(tensor);
    } finally {
      arena.releaseAll();
    }
  }

  DataType get dataType => DataType.fromId(FFITensor.datatype(nativePtr));

  Tensor to({
    DataType? dataType,
    Device? device,
    Layout? layout,
    MemoryFormat? memoryFormat,
    bool? requiresGrad,
    bool? pinnedMemory,
    bool copy = false,
    bool nonblocking = false,
  }) {
    final arena = ffi.Arena();
    try {
      final options = CTensorOptions.make(
        dataType: dataType,
        device: device,
        layout: layout,
        memoryFormat: memoryFormat,
        requiresGrad: requiresGrad,
        pinnedMemory: pinnedMemory,
        allocator: arena,
      );
      final tensor = FFITensor.to(nativePtr, options.ref, nonblocking, copy);
      return Tensor(tensor);
    } finally {
      arena.releaseAll();
    }
  }

  void to_({
    DataType? dataType,
    Device? device,
    Layout? layout,
    MemoryFormat? memoryFormat,
    bool? requiresGrad,
    bool? pinnedMemory,
    bool copy = false,
    bool nonblocking = false,
  }) {
    final arena = ffi.Arena();
    try {
      final options = CTensorOptions.make(
        dataType: dataType,
        device: device,
        layout: layout,
        memoryFormat: memoryFormat,
        requiresGrad: requiresGrad,
        pinnedMemory: pinnedMemory,
        allocator: arena,
      );
      final newTensorPtr = FFITensor.to(
        nativePtr,
        options.ref,
        nonblocking,
        copy,
      );

      if (nativePtr == newTensorPtr) return;

      // Release the old tensor
      if (shouldDelete) {
        _finalizer.detach(this);
        FFITensor.deleteTensor(nativePtr);
      }

      // Update to the new tensor
      nativePtr = newTensorPtr;

      // Attach finalizer to the new tensor
      if (shouldDelete) {
        _finalizer.attach(this, nativePtr, detach: this);
      }
    } finally {
      arena.releaseAll();
    }
  }

  void copy_(Tensor other, {bool nonBlocking = false}) {
    FFITensor.copy_(nativePtr, other.nativePtr, nonBlocking);
  }

  Tensor contiguous({MemoryFormat format = MemoryFormat.contiguous}) {
    final tensor = FFITensor.contiguous(nativePtr, format.id);
    return Tensor(tensor);
  }

  /// Returns the transposed version of the tensor. Swaps [dim0] and [dim1].
  Tensor transpose(int dim0, int dim1) {
    final tensor = FFITensor.transpose(nativePtr, dim0, dim1);
    return Tensor(tensor);
  }

  Tensor clone({MemoryFormat? memoryFormat}) {
    final arena = ffi.Arena();
    try {
      ffi.Pointer<ffi.Int8> memoryFormatPtr = ffi.nullptr;
      if (memoryFormat != null) {
        memoryFormatPtr = arena.allocate<ffi.Int8>(ffi.sizeOf<ffi.Int8>());
        memoryFormatPtr.value = memoryFormat.id;
      }
      final tensor = FFITensor.clone(nativePtr, memoryFormatPtr);
      return Tensor(tensor);
    } finally {
      arena.releaseAll();
    }
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

      final tensor = FFITensor.pad(
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

  /// [atol] is absolute tolerance. [rtol] is relative tolerance.
  bool allClose(
    Tensor other, {
    double rtol = 1e-05,
    double atol = 1e-08,
    bool equalNan = false,
  }) {
    if (!ListEquality().equals(shape, other.shape)) {
      return false;
    }
    return FFITensor.allClose(nativePtr, other.nativePtr, rtol, atol, equalNan);
  }

  Tensor operator +(dynamic /* Tensor | num */ other) {
    final arena = ffi.Arena();
    try {
      if (other is Tensor) {
        final alpha = CScalar.allocate(arena);
        alpha.ref.setInt(1);
        final tensor = FFITensor.addition(
          nativePtr,
          other.nativePtr,
          alpha.ref,
        );
        return Tensor(tensor);
      } else if (other is num) {
        var scalarTensor = Tensor.from(
          [other.toDouble()],
          [1],
          datatype: dataType,
        );
        if (device != Device.cpu) {
          scalarTensor = scalarTensor.to(device: device);
        }
        final alpha = CScalar.allocate(arena);
        alpha.ref.setInt(1);
        final tensor = FFITensor.addition(
          nativePtr,
          scalarTensor.nativePtr,
          alpha.ref,
        );
        return Tensor(tensor);
      } else if (other is (Tensor, dynamic)) {
        final alpha = CScalar.allocate(arena);
        alpha.ref.setValue(other.$2);
        final tensor = FFITensor.addition(
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

  List<int>? allCloseSlow(Tensor other, {double atol = 1e-08}) {
    if (isScalar) {
      if (!other.isScalar) return [];
      if ((scalar - other.scalar).abs() > atol) {
        print(
          'Scalar mismatch: $scalar vs ${other.scalar} diff: ${(scalar - other.scalar).abs()}',
        );
        return [];
      }
      return null;
    }

    final shape = this.shape;
    if (shape.first != other.shape.first) return [];

    for (int i = 0; i < shape.first; i++) {
      final result = this[i].allCloseSlow(other[i], atol: atol);
      if (result != null) return [i, ...result];
    }
    return null;
  }

  Tensor operator -(dynamic /* Tensor | num */ other) {
    final arena = ffi.Arena();
    try {
      if (other is Tensor) {
        final alpha = CScalar.allocate(arena);
        alpha.ref.setInt(1);
        final tensor = FFITensor.subtraction(
          nativePtr,
          other.nativePtr,
          alpha.ref,
        );
        return Tensor(tensor);
      } else if (other is num) {
        final alpha = CScalar.allocate(arena);
        alpha.ref.setInt(1);

        var scalarTensor = Tensor.from(
          [other.toDouble()],
          [1],
          datatype: dataType,
        );
        if (device != Device.cpu) {
          scalarTensor = scalarTensor.to(device: device);
        }
        final tensor = FFITensor.subtraction(
          nativePtr,
          scalarTensor.nativePtr,
          alpha.ref,
        );
        return Tensor(tensor);
      } else if (other is (Tensor, dynamic)) {
        final alpha = CScalar.allocate(arena);
        alpha.ref.setValue(other.$2);
        final tensor = FFITensor.subtraction(
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
        final tensor = FFITensor.multiplication(nativePtr, other.nativePtr);
        return Tensor(tensor);
      } else if (other is num) {
        var scalarTensor = Tensor.from(
          [other.toDouble()],
          [1],
          datatype: dataType,
        );
        if (device != Device.cpu) {
          scalarTensor = scalarTensor.to(device: device);
        }
        final tensor = FFITensor.multiplication(
          nativePtr,
          scalarTensor.nativePtr,
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

  Tensor operator /(dynamic /* Tensor | num */ other) {
    final arena = ffi.Arena();
    try {
      if (other is Tensor) {
        final tensor = FFITensor.division(nativePtr, other.nativePtr);
        return Tensor(tensor);
      } else if (other is num) {
        final scalar = CScalar.allocate(arena);
        scalar.ref.setValue(other);
        final tensor = FFITensor.divisionScalar(nativePtr, scalar.ref);
        return Tensor(tensor);
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

  Tensor pow(dynamic exponent) {
    final arena = ffi.Arena();
    try {
      final exponentScalar = CScalar.allocate(arena);
      exponentScalar.ref.setValue(exponent);
      final tensor = FFITensor.pow(nativePtr, exponentScalar.ref);
      return Tensor(tensor);
    } finally {
      arena.releaseAll();
    }
  }

  Tensor rsqrt() {
    final tensor = FFITensor.rsqrt(nativePtr);
    return Tensor(tensor);
  }

  Tensor sin() {
    final tensor = FFITensor.sin(nativePtr);
    return Tensor(tensor);
  }

  Tensor cos() {
    final tensor = FFITensor.cos(nativePtr);
    return Tensor(tensor);
  }

  Tensor exp() {
    final tensor = FFITensor.exp(nativePtr);
    return Tensor(tensor);
  }

  /// Compute the matrix or vector norm
  ///
  /// Returns the matrix norm or vector norm of a given tensor.
  ///
  /// Args:
  ///   p: The order of norm. Can be a number, 'fro' for Frobenius norm,
  ///      'nuc' for nuclear norm, or inf/-inf. Default is 2 (L2 norm).
  ///      Common values:
  ///      - 1: L1 norm (sum of absolute values)
  ///      - 2: L2 norm (Euclidean norm, default)
  ///      - double.infinity: infinity norm (max absolute value)
  ///   dim: Dimensions to compute norm over. If null, computes norm over all dimensions
  ///   keepDim: Whether to keep the reduced dimensions
  ///
  /// Returns:
  ///   Tensor containing the norm values
  ///
  /// Example:
  /// ```dart
  /// final t = Tensor.from([3.0, 4.0], [2], datatype: DataType.float32);
  /// final l2 = t.norm(2); // sqrt(3^2 + 4^2) = 5.0
  /// final l1 = t.norm(1); // |3| + |4| = 7.0
  ///
  /// final matrix = Tensor.from([1.0, 2.0, 3.0, 4.0], [2, 2], datatype: DataType.float32);
  /// final colNorms = matrix.norm(2, dim: [0]); // Norm of each column
  /// ```
  Tensor norm(num p, {List<int>? dim, bool keepDim = false}) {
    final arena = ffi.Arena();
    try {
      ffi.Pointer<ffi.Int64> dimPointer = ffi.nullptr;
      int dimLen = 0;
      if (dim != null) {
        dimLen = dim.length;
        dimPointer = arena.allocate<ffi.Int64>(
          ffi.sizeOf<ffi.Int64>() * dim.length,
        );
        dimPointer.asTypedList(dim.length).setAll(0, dim);
      }

      final scalar = CScalar.allocate(arena);
      scalar.ref.setValue(p);

      final tensor = FFITensor.norm(
        nativePtr,
        scalar.ref,
        dimPointer,
        dimLen,
        keepDim,
      );
      return Tensor(tensor);
    } finally {
      arena.releaseAll();
    }
  }

  Tensor bitwiseNot() {
    final tensor = FFITensor.bitwiseNot(nativePtr);
    return Tensor(tensor);
  }

  /// Returns the indices of the maximum value of all elements in the input tensor.
  ///
  /// Args:
  ///   dim: The dimension to reduce. If null, the argmax of the flattened input is returned.
  ///   keepDim: Whether the output tensor has dim retained or not.
  ///
  /// Returns:
  ///   A new tensor containing the indices of the maximum values.
  Tensor argmax({int? dim, bool keepDim = false}) {
    final arena = ffi.Arena();
    try {
      ffi.Pointer<ffi.Int64> dimPtr = ffi.nullptr;
      if (dim != null) {
        dimPtr = arena.allocate<ffi.Int64>(ffi.sizeOf<ffi.Int64>());
        dimPtr.value = dim;
      }
      final tensor = FFITensor.argmax(nativePtr, dimPtr, keepDim);
      return Tensor(tensor);
    } finally {
      arena.releaseAll();
    }
  }

  (Tensor values, Tensor indices) topk(
    int k, {
    int dim = -1,
    bool largest = true,
    bool sorted = true,
  }) {
    final ptr = FFITensor.topk(nativePtr, k, dim, largest, sorted);
    final values = Tensor(ptr[0]);
    final indices = Tensor(ptr[1]);
    ffi.calloc.free(ptr);
    return (values, indices);
  }

  (Tensor values, Tensor indices) sort({
    int dim = -1,
    bool descending = false,
  }) {
    final ptr = FFITensor.sort(nativePtr, dim, descending);
    final values = Tensor(ptr[0]);
    final indices = Tensor(ptr[1]);
    ffi.calloc.free(ptr);
    return (values, indices);
  }

  Tensor cumsum(int dim, {DataType? dtype}) {
    final arena = ffi.Arena();
    try {
      ffi.Pointer<ffi.Uint8> dtypePtr = ffi.nullptr;
      if (dtype != null) {
        dtypePtr = arena.allocate<ffi.Uint8>(ffi.sizeOf<ffi.Uint8>());
        dtypePtr.value = dtype.type;
      }
      final tensor = FFITensor.cumsum(nativePtr, dim, dtypePtr);
      return Tensor(tensor);
    } finally {
      arena.releaseAll();
    }
  }

  Tensor multinomial(
    int numSamples, {
    bool replacement = false,
    Generator? generator,
  }) {
    final tensor = FFITensor.multinomial(
      nativePtr,
      numSamples,
      replacement,
      generator?.nativePtr ?? ffi.nullptr,
    );
    return Tensor(tensor);
  }

  Tensor lt(dynamic other) {
    if (other is Tensor) {
      final tensor = FFITensor.ltTensor(nativePtr, other.nativePtr);
      return Tensor(tensor);
    } else {
      final arena = ffi.Arena();
      try {
        final scalar = CScalar.allocate(arena);
        scalar.ref.setValue(other);
        final tensor = FFITensor.lt(nativePtr, scalar.ref);
        return Tensor(tensor);
      } finally {
        arena.releaseAll();
      }
    }
  }

  Tensor gt(dynamic other) {
    if (other is Tensor) {
      final tensor = FFITensor.gtTensor(nativePtr, other.nativePtr);
      return Tensor(tensor);
    } else {
      final arena = ffi.Arena();
      try {
        final scalar = CScalar.allocate(arena);
        scalar.ref.setValue(other);
        final tensor = FFITensor.gt(nativePtr, scalar.ref);
        return Tensor(tensor);
      } finally {
        arena.releaseAll();
      }
    }
  }

  Tensor eq(dynamic other) {
    if (other is Tensor) {
      final tensor = FFITensor.eqTensor(nativePtr, other.nativePtr);
      return Tensor(tensor);
    } else {
      final arena = ffi.Arena();
      try {
        final scalar = CScalar.allocate(arena);
        scalar.ref.setValue(other);
        final tensor = FFITensor.eq(nativePtr, scalar.ref);
        return Tensor(tensor);
      } finally {
        arena.releaseAll();
      }
    }
  }

  Tensor maskedFill(Tensor mask, dynamic value) {
    final arena = ffi.Arena();
    try {
      final scalar = CScalar.allocate(arena);
      scalar.ref.setValue(value);
      final tensor = FFITensor.maskedFill(
        nativePtr,
        mask.nativePtr,
        scalar.ref,
      );
      return Tensor(tensor);
    } finally {
      arena.releaseAll();
    }
  }

  Tensor squeeze({int? dim}) {
    final arena = ffi.Arena();
    try {
      ffi.Pointer<ffi.Int64> dimPtr = ffi.nullptr;
      if (dim != null) {
        dimPtr = arena.allocate<ffi.Int64>(ffi.sizeOf<ffi.Int64>());
        dimPtr.value = dim;
      }
      final tensor = FFITensor.squeeze(nativePtr, dimPtr);
      return Tensor(tensor);
    } finally {
      arena.releaseAll();
    }
  }

  Tensor unsqueeze(int dim) {
    final tensor = FFITensor.unsqueeze(nativePtr, dim);
    return Tensor(tensor);
  }

  Tensor bitwiseAnd(Tensor other) {
    final tensor = FFITensor.bitwiseAnd(nativePtr, other.nativePtr);
    return Tensor(tensor);
  }

  Tensor bitwiseOr(Tensor other) {
    final tensor = FFITensor.bitwiseOr(nativePtr, other.nativePtr);
    return Tensor(tensor);
  }

  Tensor bitwiseXor(Tensor other) {
    final tensor = FFITensor.bitwiseXor(nativePtr, other.nativePtr);
    return Tensor(tensor);
  }

  Tensor sum({List<int>? dim, bool keepDim = false, DataType? dataType}) {
    final arena = ffi.Arena();
    try {
      ffi.Pointer<ffi.Int64> dimPointer = ffi.nullptr;
      int dimLen = 0;
      if (dim != null) {
        dimLen = dim.length;
        dimPointer = arena.allocate<ffi.Int64>(
          ffi.sizeOf<ffi.Int64>() * dim.length,
        );
        dimPointer.asTypedList(dim.length).setAll(0, dim);
      }

      ffi.Pointer<ffi.Uint8> dataTypePointer = ffi.nullptr;
      if (dataType != null) {
        dataTypePointer = arena.allocate<ffi.Uint8>(ffi.sizeOf<ffi.Uint8>());
        dataTypePointer.value = dataType.type;
      }

      final tensor = FFITensor.sum(
        nativePtr,
        dimPointer,
        dimLen,
        keepDim,
        dataTypePointer,
      );
      return Tensor(tensor);
    } finally {
      arena.releaseAll();
    }
  }

  Tensor mean({List<int>? dim, bool keepDim = false, DataType? dataType}) {
    final arena = ffi.Arena();
    try {
      ffi.Pointer<ffi.Int64> dimPointer = ffi.nullptr;
      int dimLen = 0;
      if (dim != null) {
        dimLen = dim.length;
        dimPointer = arena.allocate<ffi.Int64>(
          ffi.sizeOf<ffi.Int64>() * dim.length,
        );
        dimPointer.asTypedList(dim.length).setAll(0, dim);
      }

      ffi.Pointer<ffi.Uint8> dataTypePointer = ffi.nullptr;
      if (dataType != null) {
        dataTypePointer = arena.allocate<ffi.Uint8>(ffi.sizeOf<ffi.Uint8>());
        dataTypePointer.value = dataType.type;
      }

      final tensor = FFITensor.mean(
        nativePtr,
        dimPointer,
        dimLen,
        keepDim,
        dataTypePointer,
      );
      return Tensor(tensor);
    } finally {
      arena.releaseAll();
    }
  }

  Tensor matmul(Tensor other) {
    final tensor = FFITensor.matmul(nativePtr, other.nativePtr);
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
      final tensor = FFITensor.softmax(nativePtr, dim, dataTypePointer);
      return Tensor(tensor);
    } finally {
      arena.releaseAll();
    }
  }

  Tensor sigmoid() {
    final tensor = FFITensor.sigmoid(nativePtr);
    return Tensor(tensor);
  }

  Tensor relu() {
    final tensor = FFITensor.relu(nativePtr);
    return Tensor(tensor);
  }

  Tensor gelu(GeluApporimate approximate) {
    final arena = ffi.Arena();
    try {
      final activation = approximate.name.toNativeUtf8(allocator: arena);
      final tensor = FFITensor.gelu(nativePtr, activation);
      return Tensor(tensor);
    } finally {
      arena.releaseAll();
    }
  }

  Tensor silu() {
    final tensor = FFITensor.silu(nativePtr);
    return Tensor(tensor);
  }

  static Tensor cat(List<Tensor> tensors, {int dim = 0}) {
    final arena = ffi.Arena();
    try {
      final tensorsPtr = arena.allocate<CTensor>(
        ffi.sizeOf<CTensor>() * tensors.length,
      );
      for (int i = 0; i < tensors.length; i++) {
        (tensorsPtr + i).value = tensors[i].nativePtr;
      }
      final tensor = FFITensor.cat(tensorsPtr, tensors.length, dim);
      return Tensor(tensor);
    } finally {
      arena.releaseAll();
    }
  }

  /// Stack tensors along a new dimension
  ///
  /// Concatenates a sequence of tensors along a new dimension.
  /// All tensors need to be of the same size.
  ///
  /// Args:
  ///   tensors: List of tensors to stack
  ///   dim: Dimension to insert. Has to be between 0 and the number of dimensions of concatenated tensors (inclusive)
  ///
  /// Returns:
  ///   Stacked tensor with one additional dimension
  ///
  /// Example:
  /// ```dart
  /// final a = Tensor.from([1.0, 2.0], [2], datatype: DataType.float32);
  /// final b = Tensor.from([3.0, 4.0], [2], datatype: DataType.float32);
  /// final stacked = Tensor.stack([a, b], dim: 0);
  /// // stacked.shape == [2, 2]
  /// ```
  static Tensor stack(List<Tensor> tensors, {int dim = 0}) {
    final arena = ffi.Arena();
    try {
      final tensorsPtr = arena.allocate<CTensor>(
        ffi.sizeOf<CTensor>() * tensors.length,
      );
      for (int i = 0; i < tensors.length; i++) {
        (tensorsPtr + i).value = tensors[i].nativePtr;
      }
      final tensor = FFITensor.stack(tensorsPtr, tensors.length, dim);
      return Tensor(tensor);
    } finally {
      arena.releaseAll();
    }
  }

  /// Select a slice of this tensor along the given dimension at the given index
  ///
  /// This is equivalent to tensor[index] along a specific dimension.
  /// The returned tensor has the given dimension removed.
  ///
  /// Args:
  ///   dim: Dimension to slice
  ///   index: Index to select
  ///
  /// Returns:
  ///   Tensor with one less dimension
  ///
  /// Example:
  /// ```dart
  /// final t = Tensor.from([1.0, 2.0, 3.0, 4.0], [2, 2], datatype: DataType.float32);
  /// final selected = t.select(0, 1); // Select second row
  /// // selected.shape == [2]
  /// // selected == [3.0, 4.0]
  /// ```
  Tensor select(int dim, int index) {
    final tensor = FFITensor.selectDim(nativePtr, dim, index);
    return Tensor(tensor);
  }

  /// Slice this tensor along the given dimension
  ///
  /// Returns a tensor that is a narrowed version of this tensor.
  /// The dimension dim is sliced in the range [start, end) with the given step.
  ///
  /// Args:
  ///   dim: Dimension to slice
  ///   start: Starting index (inclusive)
  ///   end: Ending index (exclusive). If null, slices to the end
  ///   step: Step size (default: 1)
  ///
  /// Returns:
  ///   Sliced tensor
  ///
  /// Example:
  /// ```dart
  /// final t = Tensor.from([1.0, 2.0, 3.0, 4.0, 5.0], [5], datatype: DataType.float32);
  /// final sliced = t.slice(0, 1, 4); // [2.0, 3.0, 4.0]
  /// final stepped = t.slice(0, 0, 5, step: 2); // [1.0, 3.0, 5.0]
  /// ```
  Tensor slice(int dim, int start, {int step = 1, int? end}) {
    // PyTorch uses a very large number to represent "end of dimension"
    final int endIndex = end ?? 9223372036854775807; // max int64
    final tensor = FFITensor.slice(nativePtr, dim, start, endIndex, step);
    return Tensor(tensor);
  }

  /// Create a tensor filled with a scalar value
  ///
  /// Creates a tensor of the given shape filled with fillValue.
  ///
  /// Args:
  ///   sizes: Shape of the tensor
  ///   fillValue: Value to fill the tensor with
  ///   datatype: Data type of the tensor
  ///   device: Device to create the tensor on
  ///   layout: Memory layout
  ///   memoryFormat: Memory format
  ///   requiresGrad: Whether to track gradients
  ///   pinnedMemory: Whether to use pinned memory
  ///
  /// Returns:
  ///   Tensor filled with the specified value
  ///
  /// Example:
  /// ```dart
  /// final t = Tensor.full([2, 3], 5.0, datatype: DataType.float32);
  /// // t == [[5.0, 5.0, 5.0], [5.0, 5.0, 5.0]]
  /// ```
  static Tensor full(
    List<int> sizes,
    dynamic fillValue, {
    String? name,
    DataType? datatype,
    Device? device,
    Layout? layout,
    MemoryFormat? memoryFormat,
    bool? requiresGrad,
    bool? pinnedMemory,
  }) {
    final arena = ffi.Arena();
    try {
      final options = CTensorOptions.make(
        dataType: datatype,
        device: device,
        layout: layout,
        memoryFormat: memoryFormat,
        requiresGrad: requiresGrad,
        pinnedMemory: pinnedMemory,
        allocator: arena,
      );

      final sizesPointer = arena.allocate<ffi.Int64>(
        ffi.sizeOf<ffi.Int64>() * sizes.length,
      );
      sizesPointer.asTypedList(sizes.length).setAll(0, sizes);

      final scalar = CScalar.allocate(arena);
      scalar.ref.setValue(fillValue);

      final tensor = FFITensor.full(
        sizesPointer,
        sizes.length,
        scalar.ref,
        options.ref,
      );
      return Tensor(tensor, name: name);
    } finally {
      arena.releaseAll();
    }
  }

  List<num> toList() {
    if (dataType == DataType.float32) {
      final ptr = dataPointer.cast<ffi.Float>();
      return ptr.asTypedList(numel).toList();
    } else if (dataType == DataType.int64) {
      final ptr = dataPointer.cast<ffi.Int64>();
      return ptr.asTypedList(numel).toList();
    } else if (dataType == DataType.float64) {
      final ptr = dataPointer.cast<ffi.Double>();
      return ptr.asTypedList(numel).toList();
    } else if (dataType == DataType.int32) {
      final ptr = dataPointer.cast<ffi.Int32>();
      return ptr.asTypedList(numel).toList();
    } else if (dataType == DataType.int16) {
      final ptr = dataPointer.cast<ffi.Int16>();
      return ptr.asTypedList(numel).toList();
    } else if (dataType == DataType.int8) {
      final ptr = dataPointer.cast<ffi.Int8>();
      return ptr.asTypedList(numel).toList();
    } else if (dataType == DataType.uint8) {
      final ptr = dataPointer.cast<ffi.Uint8>();
      return ptr.asTypedList(numel).toList();
    }
    throw Exception('Unsupported data type: $dataType');
  }

  static void _print1d(StringBuffer sb, int size, Tensor tensor) {
    sb.write('[');
    for (int i = 0; i < size; i++) {
      if (i > 0) sb.write(', ');
      final scalar = tensor.scalarAt(i);
      if (scalar is double) {
        sb.write(scalar.toStringAsFixed(4));
      } else {
        sb.write(scalar);
      }
      if (i == 10 && size > 20) {
        sb.write(', ..............');
        i = size - 10;
      }
    }
    sb.write(']');
  }

  static void _print2d(StringBuffer sb, int size0, int size1, Tensor tensor) {
    sb.write('[');
    for (int i = 0; i < size0; i++) {
      _print1d(sb, size1, tensor.get(i));
      if (i == 3 && size0 > 6) {
        sb.writeln(',\n ..............');
        i = size0 - 3;
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

  Slice.all() : start = 0, end = null, step = 1;
}

enum GeluApporimate { none, tanh }

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
  static const float32 = DataType(
    name: 'Float',
    type: 6,
    safetensorName: 'F32',
  );
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
    float32,
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

Tensor interpolateNearest(Tensor input, List<int> outputSize) {
  final arena = ffi.Arena();
  try {
    ffi.Pointer<ffi.Int64> outputSizePointer = arena.allocate<ffi.Int64>(
      ffi.sizeOf<ffi.Int64>() * outputSize.length,
    );
    outputSizePointer.asTypedList(outputSize.length).setAll(0, outputSize);

    final tensorPtr = FFITensor.upsampleNearest(
      input.nativePtr,
      outputSizePointer,
      outputSize.length,
    );
    return Tensor(tensorPtr);
  } finally {
    arena.releaseAll();
  }
}

Tensor interpolateNearestScale(Tensor input, List<double> outputSizeScale) {
  final arena = ffi.Arena();
  try {
    ffi.Pointer<ffi.Double> outputSizePointer = arena.allocate<ffi.Double>(
      ffi.sizeOf<ffi.Double>() * outputSizeScale.length,
    );
    outputSizePointer
        .asTypedList(outputSizeScale.length)
        .setAll(0, outputSizeScale);

    final tensorPtr = FFITensor.upsampleNearestScale(
      input.nativePtr,
      outputSizePointer,
      outputSizeScale.length,
    );
    return Tensor(tensorPtr);
  } finally {
    arena.releaseAll();
  }
}

Tensor interpolateNearestExact(Tensor input, List<int> outputSize) {
  final arena = ffi.Arena();
  try {
    ffi.Pointer<ffi.Int64> outputSizePointer = arena.allocate<ffi.Int64>(
      ffi.sizeOf<ffi.Int64>() * outputSize.length,
    );
    outputSizePointer.asTypedList(outputSize.length).setAll(0, outputSize);

    final tensorPtr = FFITensor.upsampleNearestExact(
      input.nativePtr,
      outputSizePointer,
      outputSize.length,
    );
    return Tensor(tensorPtr);
  } finally {
    arena.releaseAll();
  }
}

Tensor interpolateNearestExactScale(
  Tensor input,
  List<double> outputSizeScale,
) {
  final arena = ffi.Arena();
  try {
    ffi.Pointer<ffi.Double> outputSizePointer = arena.allocate<ffi.Double>(
      ffi.sizeOf<ffi.Double>() * outputSizeScale.length,
    );
    outputSizePointer
        .asTypedList(outputSizeScale.length)
        .setAll(0, outputSizeScale);

    final tensorPtr = FFITensor.upsampleNearestExactScale(
      input.nativePtr,
      outputSizePointer,
      outputSizeScale.length,
    );
    return Tensor(tensorPtr);
  } finally {
    arena.releaseAll();
  }
}
