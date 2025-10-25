import 'dart:io';

import 'package:yaml/yaml.dart';

void main() async {
  final file = await File(
    'torchffi/gen/Declarations-v2.8.0.yaml',
  ).readAsBytes();
  final YamlList list = loadYaml(String.fromCharCodes(file));
  List<TorchFunctionDeclaration> functions = TorchFunctionDeclaration.fromList(list);
  functions.removeWhere((function) {
    if (function.deprecated) return true;
    if (excludePrefixes.any((prefix) => function.name.startsWith(prefix))) {
      return true;
    }
    if (excludeSuffixes.any((suffix) => function.name.endsWith(suffix))) {
      return true;
    }
    if(excludeFunctions.contains(function.name)) return true;
    return false;
  });
  print(functions.length);
  /*for (final function in functions) {
    // TODO

    if(function.returns.length > 1) {
      print('${function.name}');
    }
  }*/
  final eye = functions.firstWhere((function) => function.name == 'eye');
  print(eye.toCFFISignature());
  print(eye.toCFFIImplementation());
}

Map<String, String> cTypeNameMapping = {
  'at::Tensor': 'tensor',
  'int64_t': 'int64_t',
  'at::TensorOptions': 'int',
};

class TorchFunctionDeclaration {
  final String name;
  final String operatorName;
  final String? overloadName;
  final String schemaString;
  final List<TorchFunctionArgument> arguments;
  final List<String> methodOf;
  final List<TorchFunctionReturn> returns;
  final bool inplace;
  final bool isFactoryMethod;
  final bool deprecated;

  TorchFunctionDeclaration({
    required this.name,
    required this.operatorName,
    required this.overloadName,
    required this.schemaString,
    required this.arguments,
    required this.methodOf,
    required this.returns,
    required this.inplace,
    required this.isFactoryMethod,
    required this.deprecated,
  });

  Map<String, dynamic> toJson() => {
    'name': name,
    'operator_name': operatorName,
    'overload_name': overloadName,
    'schema_string': schemaString,
    'arguments': arguments.map((arg) => arg.toJson()).toList(),
    'method_of': methodOf,
    'returns': returns.map((ret) => ret.toJson()).toList(),
    'inplace': inplace,
    'is_factory_method': isFactoryMethod,
    'deprecated': deprecated,
  };

  @override
  String toString() => name;

  String typeName(String type) {
    if (cTypeNameMapping.containsKey(type)) {
      return cTypeNameMapping[type]!;
    }
    throw UnimplementedError('Unknown type: $type');
  }

  String cFunctionName() {
    // TODO constructors should have new prefix
    return 'torchffi_$name';
  }

  String toCFFISignature({bool terminate = true}) {
    final sb = StringBuffer();
    if(returns.length == 1) {
      // TODO write correct type
      sb.write(typeName(returns.first.dynamicType));
    } else {
      sb.write('void');
    }
    sb.write(' ');
    sb.write(cFunctionName());
    sb.write('(');
    // TODO output params for multi-output
    for (int i = 0; i < arguments.length; i++) {
      final arg = arguments[i];
      sb.write(typeName(arg.dynamicType));
      sb.write(' ');
      sb.write(arg.name);
      sb.write(', ');
    }
    // TODO write arguments
    sb.write(')');
    if(terminate) {
      sb.write(';\n');
    }
    return sb.toString();
  }

  String toCFFIImplementation() {
    final sb = StringBuffer();
    sb.write(toCFFISignature(terminate: false));
    sb.write(' {\n');
    // TODO write return type
    // TODO
    sb.write('}\n');
    return sb.toString();
  }

  static TorchFunctionDeclaration fromMap(Map map) {
    return TorchFunctionDeclaration(
      name: map['name'],
      operatorName: map['operator_name'],
      overloadName: map['overload_name'],
      schemaString: map['schema_string'],
      arguments: (map['arguments'] as List)
          .map((e) => TorchFunctionArgument.fromMap(e))
          .toList(),
      methodOf: (map['method_of'] as List).map((e) => e as String).toList(),
      returns: TorchFunctionReturn.fromList(map['returns']),
      inplace: map['inplace'],
      isFactoryMethod: map['is_factory_method'],
      deprecated: map['deprecated'],
    );
  }

  static List<TorchFunctionDeclaration> fromList(List list) {
    return list.map((e) => TorchFunctionDeclaration.fromMap(e)).toList();
  }
}

class TorchFunctionArgument {
  final String name;
  final String type;
  final String dynamicType;
  final bool isNullable;
  final dynamic def;

  TorchFunctionArgument({
    required this.name,
    required this.type,
    required this.dynamicType,
    required this.isNullable,
    required this.def,
  });

  static TorchFunctionArgument fromMap(Map map) {
    return TorchFunctionArgument(
      name: map['name'],
      type: map['type'],
      dynamicType: map['dynamic_type'],
      isNullable: map['is_nullable'],
      def: map['default'],
    );
  }

  Map<String, dynamic> toJson() => {
    'name': name,
    'type': type,
    'dynamic_type': dynamicType,
    'is_nullable': isNullable,
    'default': def.toString(),
  };
}

class TorchFunctionReturn {
  final String name;
  final String type;
  final String dynamicType;

  TorchFunctionReturn({
    required this.name,
    required this.type,
    required this.dynamicType,
  });

  static List<TorchFunctionReturn> fromList(List list) {
    return list.map((e) => TorchFunctionReturn.fromMap(e)).toList();
  }

  Map<String, dynamic> toJson() => {
    'name': name,
    'type': type,
    'dynamic_type': dynamicType,
  };

  static TorchFunctionReturn fromMap(Map map) {
    return TorchFunctionReturn(
      name: map['name'],
      type: map['type'],
      dynamicType: map['dynamic_type'],
    );
  }
}

const List<String> excludePrefixes = [
  "_thnn_",
  "_th_",
  "thnn_",
  "th_",
  "_foreach",
  "_amp_foreach",
  "_nested_tensor",
  "_fused_adam",
  "_fused_adagrad",
  "sym_",
  "_fused_sgd",
];

const List<String> excludeSuffixes = ["_forward", "_forward_out"];

const List<String> excludeFunctions = [
  "multi_margin_loss",
  "multi_margin_loss_out",
  "log_softmax_backward_data",
  "softmax_backward_data",
  "clone",
  "copy",
  "copy_out",
  "copy_",
  "conv_transpose2d_backward_out",
  "conv_transpose3d_backward_out",
  "slow_conv_transpose2d_backward_out",
  "slow_conv_transpose3d_backward_out",
  "slow_conv3d_backward_out",
  "normal",
  "_cufft_set_plan_cache_max_size",
  "_cufft_clear_plan_cache",
  "backward",
  "_amp_non_finite_check_and_unscale_",
  "_cummin_helper",
  "_cummax_helper",
  "retain_grad",
  "_validate_sparse_coo_tensor_args",
  "_sparse_semi_structured_addmm",
  "_backward",
  "size",
  "stride",
  "_assert_async",
  "gradient",
  "linalg_vector_norm",
  "linalg_vector_norm_out",
  "linalg_matrix_norm",
  "linalg_matrix_norm_out",
  "normal_out",
  "bernoulli_out",
  "nested_tensor",
  "arange_out",
];
