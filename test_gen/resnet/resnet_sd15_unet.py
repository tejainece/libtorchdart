import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from diffusers.models.resnet import ResnetBlock2D

device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = "models/diffusion/v1-5-pruned-emaonly.safetensors"
pipe = StableDiffusionPipeline.from_single_file(
    model_path,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
)
resnet: ResnetBlock2D = pipe.to(device).unet.down_blocks[0].resnets[0]
#print(resnet)

batch_size = 1
in_channels = resnet.in_channels
height, width = 64, 64

torch.manual_seed(41)
input = torch.randn(
    batch_size, in_channels, height, width,
    dtype=torch.float16 if device == "cuda" else torch.float32,
    device=device
)
time_emb_dim = 1280
temb = torch.randn(
    batch_size, time_emb_dim,
    dtype=torch.float16 if device == "cuda" else torch.float32,
    device=device
)

output = resnet(input, temb)
# print(output)

name = "unet"
tensors = {
    f"{name}.input": input,
    f"{name}.temb": temb,
    f"{name}.output": output,
    **{f"{name}.resnet.{k}": v for k, v in resnet.state_dict().items()},
}
#print(tensors.keys())

import os
os.makedirs("test_data/resnet", exist_ok=True)

from safetensors.torch import save_file
save_file(tensors, "test_data/resnet/resnet_sd15_unet_tests.safetensors")

print("\n✓ Successfully generated ResnetBlock2D unet testcases")

