import 'package:test/test.dart';
import 'package:libtorchdart/libtorchdart.dart';
import 'package:libtorchdart/src/unet/cross_attention_up2d.dart';
import 'package:libtorchdart/src/unet/transformer_2d.dart';

void main() {
  test('UNet2DConditionModel instantiation and forward pass', () {
    // Create dummy sub-modules
    final convIn = Conv2D.make(
      numInChannels: 4,
      numOutChannels: 320,
      padding: SymmetricPadding2D.same(1),
    );
    final timeProj = Timesteps(320);
    final timeEmbedding = TimestepEmbedding(
      linear1: LinearLayer(weight: Tensor.randn([1280, 320])),
      linear2: LinearLayer(weight: Tensor.randn([1280, 1280])),
      act: Activation.silu,
    );

    final downBlocks = <UNet2DDownBlock>[
      CrossAttnDownBlock2D(
        resnets: [ResnetBlock2D.make(numInChannels: 320, numOutChannels: 320)],
        attentions: [Transformer2DModel()],
        downSamplers: [Downsample2D.make(numChannels: 320)],
      ),
    ];

    final midBlock = UNet2DMidBlock(
      ResnetBlock2D.make(numInChannels: 320, numOutChannels: 320),
      resnets: [],
    );

    final upBlocks = <UNet2DUpBlock>[
      CrossAttnUpBlock2D(
        resnets: [
          ResnetBlock2D.make(numInChannels: 640, numOutChannels: 320),
        ], // 320 from down + 320 from up
        attentions: [Transformer2DModel()],
        upsamplers: [Upsample2D.make(numChannels: 320)],
      ),
    ];

    final convNormOut = GroupNorm.make(numGroups: 32, numChannels: 320);
    final convAct = Activation.silu;
    final convOut = Conv2D.make(
      numInChannels: 320,
      numOutChannels: 4,
      padding: SymmetricPadding2D.same(1),
    );

    /* TODO
    final model = UNet2DConditionModel(
      sampleSize: 64,
      inChannels: 4,
      outChannels: 4,
      centerInputSample: false,
      flipSinCosToFloat32: true,
      freqShift: 0,
      blockOutChannels: [320],
      layersPerBlock: 1,
      downsamplePadding: 1,
      midBlockScaleFactor: 1.0,
      actFn: "silu",
      normNumGroups: 32,
      normEps: 1e-5,
      crossAttentionDim: 768,
      attentionHeadDim: 8,
      convIn: convIn,
      timeProj: timeProj,
      timeEmbedding: timeEmbedding,
      downBlocks: downBlocks,
      midBlock: midBlock,
      upBlocks: upBlocks,
      convNormOut: convNormOut,
      convAct: convAct,
      convOut: convOut,
    );

    final sample = Tensor.randn([1, 4, 64, 64]);
    final timestep = 1;
    final encoderHiddenStates = Tensor.randn([1, 77, 768]);

    final output = model.forward(
      sample,
      timestep,
      encoderHiddenStates: encoderHiddenStates,
    );

    expect(output.shape, equals([1, 4, 64, 64]));
    */
  });
}
