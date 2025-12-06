import 'package:tensor/tensor.dart';
import 'package:universal_io/io.dart';

export 'metadata.dart';
export 'storage.dart';

abstract class SafeTensors {
  SafeTensorHeader get header;

  // TODO device type argument
  Tensor getTensor(String name);
}

/// File-backed safetensors. The tensor data is read from file/harddisk on demand.
/// Recommended when the tensors are read once into memory. Not recommended when
/// the
class SafeTensorsFile {
  final SafeTensorHeader header;
  final String path;
  final int fileLength;

  SafeTensorsFile({
    required this.header,
    required this.path,
    required this.fileLength,
  });

  SafeTensorLoader cpuLoader() {
    if (Platform.isLinux ||
        Platform.isMacOS ||
        Platform.isAndroid ||
        Platform.isIOS ||
        Platform.isFuchsia) {
      return MmapSafeTensorLoader.make(
        header: header,
        path: path,
        fileLength: fileLength,
      );
    }
    return FileIOSafeTensorLoader(header: header);
  }

  SafeTensorLoader cudaLoader() {
    // TODO if GDS is supported use GDS
    if (false) {
      return CudaGDSSafeTensorLoader(header: header);
    }
    return cpuLoader();
  }

  MmapSafeTensorLoader mmapTensorLoader() => MmapSafeTensorLoader.make(
    header: header,
    path: path,
    fileLength: fileLength,
  );

  Tensor getTensor(String name) {
    // TODO read tensor
    throw UnimplementedError();
  }

  static Future<SafeTensorsFile> load(String path) async {
    RandomAccessFile file = await File(path).open();
    final fileLength = await file.length();
    try {
      final header = await SafeTensorHeader.read(file);
      return SafeTensorsFile(
        header: header,
        path: path,
        fileLength: fileLength,
      );
    } finally {
      await file.close();
    }
  }
}
