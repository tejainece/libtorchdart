from diffusers.models.unets.unet_2d_blocks import DownEncoderBlock2D
import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

dtype = torch.float32

model_path = "models/diffusion/v1-5-pruned-emaonly.safetensors"
pipe = StableDiffusionPipeline.from_single_file(
    model_path,
    torch_dtype=dtype,
)
downEncoder: DownEncoderBlock2D = pipe.to(device).vae.encoder.down_blocks[0]

# Input shape for the first block: (batch_size, in_channels, height, width)
# From inspection: Block 0 has 128 in_channels
input = torch.randn(1, 128, 64, 64).to(device).to(dtype=dtype)

output = downEncoder.forward(input)
#print(output)

name = "vae"
tensors = {
    f"{name}.input": input,
    f"{name}.output": output,
    **{f"{name}.block.{k}": v for k, v in downEncoder.state_dict().items()},
}
#print(tensors.keys())
metadata = {
    f"{name}.in_channels": str(downEncoder.resnets[0].in_channels),
    f"{name}.out_channels": str(downEncoder.resnets[0].out_channels),
    f"{name}.num_layers": str(len(downEncoder.resnets)),
    f"{name}.resnet_eps": str(downEncoder.resnets[0].norm1.eps),  # pyright: ignore[reportAttributeAccessIssue]
    f"{name}.resnet_act_fn": str(downEncoder.resnets[0].nonlinearity.__class__.__name__),
    f"{name}.resnet_groups": str(downEncoder.resnets[0].norm1.num_groups),  # pyright: ignore[reportAttributeAccessIssue]
    f"{name}.output_scale_factor": str(downEncoder.resnets[0].output_scale_factor),
    f"{name}.add_downsample": str(len(downEncoder.downsamplers) > 0),  # pyright: ignore[reportArgumentType]
    f"{name}.downsample_padding": str(downEncoder.downsamplers[0].padding if len(downEncoder.downsamplers) > 0 else 0),  # pyright: ignore[reportArgumentType, reportOptionalSubscript]
}

import os
os.makedirs("test_data/vae/down_encoder", exist_ok=True)

from safetensors.torch import save_file
save_file(tensors, "test_data/vae/down_encoder/downencoder_vae.safetensors", metadata=metadata)

print("\n✓ Successfully generated DownEncoderBlock2D vae testcases")