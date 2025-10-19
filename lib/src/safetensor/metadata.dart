

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

class SafeTensorInfo {
  final String dtype;
  final List<int> shape;
  final int startOffset;
  final int endOffset;

  SafeTensorInfo({
    required this.dtype,
    required this.shape,
    required this.startOffset,
    required this.endOffset,
  });

  int get bytes => endOffset - startOffset;

  static SafeTensorInfo fromMap(Map map) => SafeTensorInfo(
    dtype: map['dtype'],
    shape: (map['shape'] as List).cast<int>(),
    startOffset: map['data_offsets'][0],
    endOffset: map['data_offsets'][1],
  );

  static Map<String, SafeTensorInfo> fromMapOfMap(Map map) =>
      map.map<String, SafeTensorInfo>(
        (k, v) => MapEntry(k, SafeTensorInfo.fromMap(v)),
      );

  static List<SafeTensorInfo> fromList(List list) =>
      list.map((e) => SafeTensorInfo.fromMap(e)).toList();

  Map<String, dynamic> toJson() => {
    'dtype': dtype,
    'shape': shape,
    'data_offsets': [startOffset, endOffset],
  };
}

class SafeTensorHeader {
  final Map<String, String> metadata;
  final Map<String, SafeTensorInfo> tensorInfos;
  final int dataOffset;

  SafeTensorHeader({required this.metadata, required this.tensorInfos, required this.dataOffset});

  static Future<SafeTensorHeader> read(RandomAccessFile file) async {
    await file.setPosition(0);
    final headerLenBuffer = Uint8List(8);
    final headerLengthReadCount = await file.readInto(
      headerLenBuffer,
      0,
      headerLenBuffer.length,
    );
    if (headerLengthReadCount != 8) {
      throw Exception(
        'expected to read 8 bytes but got $headerLengthReadCount bytes',
      );
    }
    int headerLen = ByteData.sublistView(
      headerLenBuffer,
    ).getUint64(0, Endian.little);
    final buffer = Uint8List(headerLen);
    final headerBytesCount = await file.readInto(buffer, 0, buffer.length);
    if (headerBytesCount != headerLen) {
      throw Exception(
        'expected to read ${headerLen - 8} bytes but got $headerBytesCount bytes',
      );
    }
    final headerJson = utf8.decode(buffer);
    final Map map = json.decode(headerJson);
    final metadata = (map.remove('__metadata__') ?? {}).cast<String, String>();
    final tensorMap = SafeTensorInfo.fromMapOfMap(map);
    return SafeTensorHeader(metadata: metadata, tensorInfos: tensorMap, dataOffset: 8 + headerLen);
  }
}
