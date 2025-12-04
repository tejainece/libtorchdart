import torch
from diffusers.models.resnet import ResnetBlock2D

torch_device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(0)
sample = torch.randn(1, 32, 64, 64).to(torch_device)
temb = torch.randn(1, 128).to(torch_device)
resnet = ResnetBlock2D(in_channels=32, temb_channels=128).to(torch_device)
output = resnet.forward(sample, temb)

name = "simple1"
tensors = {
    f"{name}.input": sample,
    f"{name}.temb": temb,
    f"{name}.output": output,
    **{f"{name}.resnet.{k}": v for k, v in resnet.state_dict().items()},
}
#print(tensors.keys())

import os
os.makedirs("test_data/resnet", exist_ok=True)

from safetensors.torch import save_file
save_file(tensors, "test_data/resnet/resnet_tests.safetensors")

print("\n✓ Successfully generated ResnetBlock2D testcases")