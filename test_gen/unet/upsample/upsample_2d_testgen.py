import torch
from diffusers.models.upsampling import Upsample2D
import os
from safetensors.torch import save_file

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

channels = 32
use_conv = False
out_channels = 32
padding = 1

upsample = Upsample2D(
    channels=channels,
    use_conv=use_conv,
    out_channels=out_channels,
    padding=padding,
).to(device)

input = torch.randn(1, channels, 32, 32).to(device)

output = upsample(input)

name = "upsample"
tensors = {
    f"{name}.input": input,
    f"{name}.output": output,
    **{f"{name}.upsample.{k}": v for k, v in upsample.state_dict().items()}
}
metadata = {
    f"{name}.padding": str(upsample.conv.padding if upsample.conv else 0),
}

output_dir = "test_data/unet/upsample"
os.makedirs(output_dir, exist_ok=True)

save_path = os.path.join(output_dir, "upsample_simple.safetensors")
save_file(tensors, save_path, metadata=metadata)

print(f"\n✓ Successfully generated Upsample2D testcases")
