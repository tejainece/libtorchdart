import 'package:libtorchdart/libtorchdart.dart';
import 'package:libtorchdart/src/safetensor/storage.dart';

abstract class DiffusionPipeline {}

abstract class SimpleDiffusionPipeline {}

abstract class TextEncoder {}

class StableDiffusion implements SimpleDiffusionPipeline {
  final Tokenizer tokenizer;
  final TextEncoder textEncoder;

  StableDiffusion({required this.tokenizer, required this.textEncoder});

  // TODO encode prompt
  // TODO encode image
  // TODO ip adapter
  // TODO lora
  // TODO controlnet

  Future<void> forward(
    DiffusionInput input /* TODO can this be multiple?*/, {
    int? width,
    int? height,
    int numInterefenceSteps = 50,
    List<int>? timeSteps,
    List<double>? sigmas,
    double guidanceScale = 7.5,
    int numImagesPerPrompt = 1,
    // TODO IP adapter image
    // TODO IP adapter image embeds
    // TODO guidance rescale
  }) async {
    // TODO implement batching
    // TODO adjust lora scale

    Tensor? promptEmbeddings = input.promptEmbeddings;
    if (promptEmbeddings == null) {
      // TODO handle textual inversion
      // TODO
    }

    // TODO
  }

  static Future<StableDiffusion> loadFromSafeTensor(
    SafeTensorLoader loader,
  ) async {
    final tokenizer = await CLIPTokenizer.loadFromFile(
      'models/tokenizer/'
    );
    final textEncoder = await ClipTextTransformer.loadFromSafeTensor(
      loader,
      prefix: 'cond_stage_model.transformer.text_model.',
      config: ClipTextConfig.v1_5 /* TODO */,
    );
    // TODO
    return StableDiffusion(textEncoder: textEncoder);
  }
}

class DiffusionInput {
  final String? prompt;
  final String? negativePrompt;
  final Tensor? latent;
  final Tensor? promptEmbeddings;
  final Tensor? negativePromptEmbeddings;

  DiffusionInput({
    this.prompt,
    this.negativePrompt,
    this.latent,
    this.promptEmbeddings,
    this.negativePromptEmbeddings,
  });
}
