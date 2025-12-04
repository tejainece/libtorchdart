import 'package:libtorchdart/libtorchdart.dart';

class GPT2LMHeadModel extends Module {
  final GPT2Model transformer;
  final LinearLayer lmHead;

  GPT2LMHeadModel({
    required super.name,
    required this.transformer,
    required this.lmHead,
  });

  @override
  Tensor forward(
    Tensor inputIds, {
    Tensor? pastKeyValues,
    Tensor? attentionMask,
    Tensor? tokenTypeIds,
    Tensor? positionIds,
    Tensor? headMask,
    Tensor? inputsEmbeds,
    Tensor? encoderHiddenStates,
    Tensor? encoderAttentionMask,
    Tensor? labels,
    bool useCache = false,
    bool outputAttentions = false,
    bool outputHiddenStates = false,
    required Context context,
  }) {
    context.onloadModule(this);

    Tensor hiddenStates = transformer.forward(
      inputIds,
      pastKeyValues: pastKeyValues,
      attentionMask: attentionMask,
      tokenTypeIds: tokenTypeIds,
      positionIds: positionIds,
      headMask: headMask,
      inputsEmbeds: inputsEmbeds,
      encoderHiddenStates: encoderHiddenStates,
      encoderAttentionMask: encoderAttentionMask,
      useCache: useCache,
      outputAttentions: outputAttentions,
      outputHiddenStates: outputHiddenStates,
      context: context,
    );

    Tensor lmLogits = lmHead.forward(hiddenStates, context: context);

    // TODO: Calculate loss if labels are provided

    return lmLogits;
  }

  @override
  void resetParameters() {
    transformer.resetParameters();
    lmHead.resetParameters();
  }

  @override
  late final Iterable<Tensor> parameters = [
    ...transformer.parameters,
    ...lmHead.parameters,
  ];

  @override
  Map<String, dynamic> get meta => {};

  @override
  late final Iterable<Module> submodules = [transformer, lmHead];

  static GPT2LMHeadModel make({
    required GPT2Config config,
    required String name,
  }) {
    final transformer = GPT2Model.make(config: config, name: 'transformer');

    final lmHead = LinearLayer.make(
      name: 'lm_head',
      inFeatures: config.nEmbd,
      outFeatures: config.vocabSize,
      hasBias: false,
    );

    // Tie weights
    // Note: In LibTorch/PyTorch, weight tying is usually done by sharing the underlying tensor.
    // Here we might need to manually set lmHead.weight to transformer.wte.weights
    // But LinearLayer.make creates a new weight tensor.
    // So we should probably construct LinearLayer manually or overwrite the weight.

    // For now, let's just create it. Tying weights might require a specific method or manual assignment.
    // If we want to tie weights:
    // lmHead.weight = transformer.wte.weights;
    // But `weight` is final in LinearLayer.
    // So we would need to create LinearLayer with the existing tensor.

    return GPT2LMHeadModel(
      name: name,
      transformer: transformer,
      lmHead: lmHead,
    );
  }
}
