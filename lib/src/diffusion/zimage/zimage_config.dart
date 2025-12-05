import 'package:libtorchdart/src/transformers/s3dit/s3dit_config.dart';

/// Configuration for Z-Image diffusion pipeline
class ZImageConfig {
  /// S3-DiT transformer configuration
  final S3DiTConfig s3ditConfig;

  /// VAE latent channels
  final int vaeLatentChannels;

  /// VAE scaling factor
  final double vaeScaleFactor;

  /// Text embedding dimension (from text encoder like Qwen3-4B)
  final int textEmbedDim;

  /// Semantic vision embedding dimension (from SigLIP)
  final int? semanticEmbedDim;

  /// Projection dimension for modality-specific MLPs
  final int projectionDim;

  /// Default image height in pixels
  final int defaultHeight;

  /// Default image width in pixels
  final int defaultWidth;

  /// VAE downsampling factor
  final int vaeDownsampleFactor;

  /// Number of inference steps
  final int numInferenceSteps;

  /// Guidance scale for classifier-free guidance
  final double guidanceScale;

  /// Scheduler type ('euler', 'ddpm', etc.)
  final String schedulerType;

  ZImageConfig({
    this.s3ditConfig = S3DiTConfig.zImageBase,
    this.vaeLatentChannels = 4,
    this.vaeScaleFactor = 0.18215,
    this.textEmbedDim = 3584, // Qwen3-4B dimension
    this.semanticEmbedDim,
    int? projectionDim,
    this.defaultHeight = 1024,
    this.defaultWidth = 1024,
    this.vaeDownsampleFactor = 8,
    this.numInferenceSteps = 50,
    this.guidanceScale = 7.5,
    this.schedulerType = 'euler',
  }) : projectionDim = projectionDim ?? s3ditConfig.hiddenSize;

  /// Get latent height from image height
  int get latentHeight => defaultHeight ~/ vaeDownsampleFactor;

  /// Get latent width from image width
  int get latentWidth => defaultWidth ~/ vaeDownsampleFactor;

  /// Get number of latent tokens
  int get numLatentTokens => latentHeight * latentWidth;

  /// Default configuration for Z-Image Base
  static final ZImageConfig zImageBase = ZImageConfig(
    s3ditConfig: S3DiTConfig.zImageBase,
    vaeLatentChannels: 4,
    vaeScaleFactor: 0.18215,
    textEmbedDim: 3584,
    defaultHeight: 1024,
    defaultWidth: 1024,
    vaeDownsampleFactor: 8,
    numInferenceSteps: 50,
    guidanceScale: 7.5,
    schedulerType: 'euler',
  );

  /// Configuration for Z-Image Turbo (fewer steps)
  static final ZImageConfig zImageTurbo = ZImageConfig(
    s3ditConfig: S3DiTConfig.zImageTurbo,
    vaeLatentChannels: 4,
    vaeScaleFactor: 0.18215,
    textEmbedDim: 3584,
    defaultHeight: 1024,
    defaultWidth: 1024,
    vaeDownsampleFactor: 8,
    numInferenceSteps: 8, // Distilled for fast inference
    guidanceScale: 7.5,
    schedulerType: 'euler',
  );

  /// Configuration for Z-Image Edit (includes semantic vision)
  static final ZImageConfig zImageEdit = ZImageConfig(
    s3ditConfig: S3DiTConfig.zImageBase,
    vaeLatentChannels: 4,
    vaeScaleFactor: 0.18215,
    textEmbedDim: 3584,
    semanticEmbedDim: 1152, // SigLIP dimension
    defaultHeight: 1024,
    defaultWidth: 1024,
    vaeDownsampleFactor: 8,
    numInferenceSteps: 50,
    guidanceScale: 7.5,
    schedulerType: 'euler',
  );
}
