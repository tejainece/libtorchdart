import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = "models/diffusion/v1-5-pruned-emaonly.safetensors"
pipe = StableDiffusionPipeline.from_single_file(
    model_path,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
)
resnet = pipe.to(device).vae.decoder.mid_block.resnets[0]
#print(resnet)

batch_size = 1
in_channels = resnet.in_channels
height, width = 8, 8

torch.manual_seed(42)
input = torch.randn(
    batch_size, in_channels, height, width,
    dtype=torch.float16 if device == "cuda" else torch.float32,
    device=device
)
output = resnet(input, temb=None)
#print(output)

name = "vae"
tensors = {
    f"{name}.input": input,
    f"{name}.output": output,
    **{f"{name}.resnet.{k}": v for k, v in resnet.state_dict().items()},
}
#print(tensors.keys())

import os
os.makedirs("test_data/resnet", exist_ok=True)

from safetensors.torch import save_file
save_file(tensors, "test_data/resnet/resnet_sd15_vae_tests.safetensors")

print("\n✓ Successfully generated ResnetBlock2D vae testcases")