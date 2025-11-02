import 'package:libtorchdart/libtorchdart.dart';
import 'package:libtorchdart/src/autoencoder/vae.dart';
import 'package:libtorchdart/src/safetensor/storage.dart';
import 'package:libtorchdart/src/unets/unet2d_conditional.dart';

abstract class DiffusionPipeline {}

abstract class SimpleDiffusionPipeline {}

abstract class TextEncoder {}

abstract class FeatureExtractor {}

abstract class Scheduler {}

abstract class SafetyChecker {}

class CLIPImageProcessor implements FeatureExtractor {}

class PNDMScheduler implements Scheduler {}

class StableDiffusionSafetyChecker implements SafetyChecker {}

class StableDiffusion implements SimpleDiffusionPipeline {
  final Tokenizer tokenizer;
  final TextEncoder textEncoder;
  final FeatureExtractor featureExtractor;
  final UNet unet;
  final Vae vae;
  final Scheduler scheduler;
  final SafetyChecker safetyChecker;

  StableDiffusion({
    required this.tokenizer,
    required this.textEncoder,
    required this.featureExtractor,
    required this.unet,
    required this.vae,
    required this.scheduler,
    required this.safetyChecker,
  });

  // TODO encode prompt
  // TODO encode image
  // TODO ip adapter
  // TODO lora
  // TODO controlnet

  Future<void> forward(
    DiffusionInput input /* TODO implement batching?*/, {
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
      'models/tokenizer/',
      config: ClipTextConfig.v1_5,
    );
    final textEncoder = await ClipTextTransformer.loadFromSafeTensor(
      loader,
      prefix: 'cond_stage_model.transformer.text_model.',
      config: ClipTextConfig.v1_5 /* TODO */,
    );
    final featureExtractor = CLIPImageProcessor();
    final unet = await UNet2DConditionModel.loadFromSafeTensor(loader);
    final vae = await AutoencoderKL.loadFromSafeTensor(loader);
    final scheduler = PNDMScheduler();
    final safetyChecker = StableDiffusionSafetyChecker();

    return StableDiffusion(
      tokenizer: tokenizer,
      textEncoder: textEncoder,
      featureExtractor: featureExtractor,
      unet: unet,
      vae: vae,
      scheduler: scheduler,
      safetyChecker: safetyChecker,
    );
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
