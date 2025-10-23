import 'dart:convert';

import 'package:universal_io/io.dart';

import 'package:libtorchdart/src/nn/activation.dart';

class CLIPConfig {
  final int vocabSize;
  final int embedDim;
  final Activation activation;
  final int intermediateSize;
  final int maxPositionEmbeddings;
  final String? padWith;
  final int numHiddenLayers;
  final int numAttentionHeads;
  final int projectionDim;

  const CLIPConfig({
    required this.vocabSize,
    required this.embedDim,
    required this.activation,
    required this.intermediateSize,
    required this.maxPositionEmbeddings,
    required this.padWith,
    required this.numHiddenLayers,
    required this.numAttentionHeads,
    required this.projectionDim,
  });

  static const CLIPConfig v1_5 = CLIPConfig(
    vocabSize: 49408,
    embedDim: 768,
    activation: Activation.quickGelu,
    intermediateSize: 3072,
    maxPositionEmbeddings: 77,
    padWith: null,
    numHiddenLayers: 12,
    numAttentionHeads: 12,
    projectionDim: 768,
  );

  static const CLIPConfig v2_1 = CLIPConfig(
    vocabSize: 49408,
    embedDim: 1024,
    activation: Activation.gelu,
    intermediateSize: 4096,
    maxPositionEmbeddings: 77,
    padWith: "!",
    numHiddenLayers: 23,
    numAttentionHeads: 16,
    projectionDim: 512,
  );
}

abstract class Tokenizer {
  List<int> encode(String input);
  String decode(List<int> tokens);
}

class CLIPTokenizer implements Tokenizer {
  final Map<String, int> encoder;
  final List<String> decoder;
  final Map<String, int> bpeRanks;
  final int startOfTextToken;
  final int endOfTextToken;
  final CLIPConfig config;

  CLIPTokenizer({
    required this.encoder,
    required this.decoder,
    required this.bpeRanks,
    required this.startOfTextToken,
    required this.endOfTextToken,
    required this.config,
  });

  /// Encodes a single given token
  List<int> encodeToken(String token) {
    List<String> syllables = token.toLowerCase().split("");
    if (syllables.isEmpty) return [];
    // mark last syllable as termination syllabal
    syllables.last = '${syllables.last}</w>';

    while (syllables.length > 1) {
      final pairs = uniqueAdjecantPairs(syllables);
      (int rank, (String first, String second))? currentMin;
      for (final pair in pairs) {
        final bpeRank = bpeRanks['${pair.$1} ${pair.$2}'];
        if (bpeRank == null) continue;
        if (currentMin != null && bpeRank > currentMin.$1) continue;
        currentMin = (bpeRank, pair);
      }

      if (currentMin == null) break;

      final (first, second) = currentMin.$2;
      final newChars = <String>[];
      var index = 0;
      while (index < syllables.length) {
        final w = syllables[index];
        if (index + 1 < syllables.length &&
            w == first &&
            syllables[index + 1] == second) {
          newChars.add('$first$second');
          index += 2;
        } else {
          newChars.add(w);
          index += 1;
        }
      }
      syllables = newChars;
    }
    return syllables.map((e) => encoder[e]!).toList();
  }

  List<int> encodeWithPad(String input, {int? padSize}) {
    input = input.toLowerCase();
    List<int> encoded = <int>[startOfTextToken];
    re.allMatches(input).forEach((match) {
      final token = match.group(0)!;
      encoded.addAll(encodeToken(token));
    });

    if (padSize != null) {
      int padWith = endOfTextToken;
      if (config.padWith != null) {
        int? padWithEncoded = encoder[config.padWith];
        if (padWithEncoded == null) {
          throw Exception('Encoding for padding not found');
        }
        padWith = padWithEncoded;
      }
      if (padSize < encoded.length) {
        encoded = encoded.sublist(0, padSize - 1);
        encoded.add(padWith);
      } else {
        encoded.add(endOfTextToken);
        for (int i = encoded.length; i < padSize; i++) {
          encoded.add(padWith);
        }
      }
    } else {
      encoded.add(endOfTextToken);
    }

    return encoded;
  }

  @override
  List<int> encode(String input) =>
      encodeWithPad(input, padSize: config.maxPositionEmbeddings);

  @override
  String decode(List<int> tokens) {
    return tokens.map((token) => decoder[token].replaceAll('</w>', ' ')).join();
  }

  static Set<(String, String)> uniqueAdjecantPairs(List<String> char) {
    final pairs = <(String, String)>{};
    for (final (i, v) in char.indexed) {
      if (i > 0) {
        pairs.add((char[i - 1], v));
      }
    }
    return pairs;
  }

  static Future<CLIPTokenizer> loadFromFile(
    String bpePath, {
    required CLIPConfig config,
  }) async {
    if (config.vocabSize != 49408) {
      throw Exception('vocabSize must be 49408');
    }

    final decoder = List<String>.filled(49408, '');
    final encoder = <String, int>{};
    for (final (i, token) in _unicodeChars.indexed) {
      decoder[i] = token;
      encoder[token] = i;
    }
    for (final (i, token) in _unicodeChars.indexed) {
      decoder[i + _unicodeChars.length] = '$token</w>';
      encoder['$token</w>'] = i + _unicodeChars.length;
    }

    final bpe = await File(bpePath).readAsString();
    final bpeRanks = <String, int>{};
    final lines = LineSplitter().convert(bpe).skip(1).take(numBpeEntries);
    for (final (i, line) in lines.indexed) {
      final parts = line.split(' ');
      if (parts.length != 2) throw Exception('invalid BPE format');
      bpeRanks['${parts[0]} ${parts[1]}'] = i;
      decoder[_unicodeChars.length * 2 + i] = '${parts[0]}${parts[1]}';
      encoder['${parts[0]}${parts[1]}'] = _unicodeChars.length * 2 + i;
    }
    for (final (i, token) in ['<|startoftext|>', '<|endoftext|>'].indexed) {
      decoder[_unicodeChars.length * 2 + numBpeEntries + i] = token;
      encoder[token] = _unicodeChars.length * 2 + numBpeEntries + i;
    }

    return CLIPTokenizer(
      encoder: encoder,
      decoder: decoder,
      bpeRanks: bpeRanks,
      startOfTextToken: encoder['<|startoftext|>']!,
      endOfTextToken: encoder['<|endoftext|>']!,
      config: config,
    );
  }

  static final RegExp re = RegExp(
    r'''<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+''',
    unicode: true,
  );

  static const int numBpeEntries = 49152 - 256 - 2;
}

const _unicodeChars = [
  '!',
  '"',
  '#',
  '\$',
  '%',
  '&',
  "'",
  '(',
  ')',
  '*',
  '+',
  ',',
  '-',
  '.',
  '/',
  '0',
  '1',
  '2',
  '3',
  '4',
  '5',
  '6',
  '7',
  '8',
  '9',
  ':',
  ';',
  '<',
  '=',
  '>',
  '?',
  '@',
  'A',
  'B',
  'C',
  'D',
  'E',
  'F',
  'G',
  'H',
  'I',
  'J',
  'K',
  'L',
  'M',
  'N',
  'O',
  'P',
  'Q',
  'R',
  'S',
  'T',
  'U',
  'V',
  'W',
  'X',
  'Y',
  'Z',
  '[',
  '\\',
  ']',
  '^',
  '_',
  '`',
  'a',
  'b',
  'c',
  'd',
  'e',
  'f',
  'g',
  'h',
  'i',
  'j',
  'k',
  'l',
  'm',
  'n',
  'o',
  'p',
  'q',
  'r',
  's',
  't',
  'u',
  'v',
  'w',
  'x',
  'y',
  'z',
  '{',
  '|',
  '}',
  '~',
  '¡',
  '¢',
  '£',
  '¤',
  '¥',
  '¦',
  '§',
  '¨',
  '©',
  'ª',
  '«',
  '¬',
  '®',
  '¯',
  '°',
  '±',
  '²',
  '³',
  '´',
  'µ',
  '¶',
  '·',
  '¸',
  '¹',
  'º',
  '»',
  '¼',
  '½',
  '¾',
  '¿',
  'À',
  'Á',
  'Â',
  'Ã',
  'Ä',
  'Å',
  'Æ',
  'Ç',
  'È',
  'É',
  'Ê',
  'Ë',
  'Ì',
  'Í',
  'Î',
  'Ï',
  'Ð',
  'Ñ',
  'Ò',
  'Ó',
  'Ô',
  'Õ',
  'Ö',
  '×',
  'Ø',
  'Ù',
  'Ú',
  'Û',
  'Ü',
  'Ý',
  'Þ',
  'ß',
  'à',
  'á',
  'â',
  'ã',
  'ä',
  'å',
  'æ',
  'ç',
  'è',
  'é',
  'ê',
  'ë',
  'ì',
  'í',
  'î',
  'ï',
  'ð',
  'ñ',
  'ò',
  'ó',
  'ô',
  'õ',
  'ö',
  '÷',
  'ø',
  'ù',
  'ú',
  'û',
  'ü',
  'ý',
  'þ',
  'ÿ',
  'Ā',
  'ā',
  'Ă',
  'ă',
  'Ą',
  'ą',
  'Ć',
  'ć',
  'Ĉ',
  'ĉ',
  'Ċ',
  'ċ',
  'Č',
  'č',
  'Ď',
  'ď',
  'Đ',
  'đ',
  'Ē',
  'ē',
  'Ĕ',
  'ĕ',
  'Ė',
  'ė',
  'Ę',
  'ę',
  'Ě',
  'ě',
  'Ĝ',
  'ĝ',
  'Ğ',
  'ğ',
  'Ġ',
  'ġ',
  'Ģ',
  'ģ',
  'Ĥ',
  'ĥ',
  'Ħ',
  'ħ',
  'Ĩ',
  'ĩ',
  'Ī',
  'ī',
  'Ĭ',
  'ĭ',
  'Į',
  'į',
  'İ',
  'ı',
  'Ĳ',
  'ĳ',
  'Ĵ',
  'ĵ',
  'Ķ',
  'ķ',
  'ĸ',
  'Ĺ',
  'ĺ',
  'Ļ',
  'ļ',
  'Ľ',
  'ľ',
  'Ŀ',
  'ŀ',
  'Ł',
  'ł',
  'Ń',
];
