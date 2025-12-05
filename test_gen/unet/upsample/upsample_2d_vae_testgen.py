import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from diffusers.models.upsampling import Upsample2D
import os
from safetensors.torch import save_file

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = "models/diffusion/v1-5-pruned-emaonly.safetensors"
pipe = StableDiffusionPipeline.from_single_file(
    model_path,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
)

up_block = pipe.vae.decoder.up_blocks[0]
upsample: Upsample2D = up_block.upsamplers[0].to(device)
print(upsample)

in_channels: int = upsample.conv.in_channels  # pyright: ignore[reportOptionalMemberAccess]

input = torch.randn(1, in_channels, 64, 64).to(device=device, dtype=torch.float16 if device == "cuda" else torch.float32)

output = upsample.forward(input)

name = "vae1"
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

save_path = os.path.join(output_dir, "upsample_vae.safetensors")
save_file(tensors, save_path, metadata=metadata)

print(f"\n✓ Successfully generated Upsample2D VAE testcases")
