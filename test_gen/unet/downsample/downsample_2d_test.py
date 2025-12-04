import torch
from diffusers.models.downsampling import Downsample2D
import os
from safetensors.torch import save_file

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

channels = 32
use_conv = True
out_channels = 32
padding = 1
# TODO also include norm

downsample = Downsample2D(
    channels=channels,
    use_conv=use_conv,
    out_channels=out_channels,
    padding=padding,
    name="op"
).to(device)
print(downsample)

input = torch.randn(1, channels, 64, 64).to(device)

output = downsample(input)

name = "downsample2d"
tensors = {
    f"{name}.input": input,
    f"{name}.output": output,
    **{f"{name}.downsample.{k}": v for k, v in downsample.state_dict().items()}
}
metadata = {
    f"{name}.padding": str(downsample.padding),
}

output_dir = "test_data/unet/downsample"
os.makedirs(output_dir, exist_ok=True)

save_path = os.path.join(output_dir, "downsample_simple.safetensors")
save_file(tensors, save_path, metadata=metadata)

print(f"\n✓ Successfully generated Downsample2D testcases")
