from diffusers.models.downsampling import Downsample2D
import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
import os
from safetensors.torch import save_file

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = "models/diffusion/v1-5-pruned-emaonly.safetensors"
pipe = StableDiffusionPipeline.from_single_file(
    model_path,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
)

down_block = pipe.vae.encoder.down_blocks[0]
downsample: Downsample2D = down_block.downsamplers[0].to(device)

print(f"Downsample block: {downsample}")

in_channels: int = downsample.conv.in_channels  # pyright: ignore[reportAssignmentType]

input = torch.randn(1, in_channels, 64, 64).to(device=device, dtype=torch.float16 if device == "cuda" else torch.float32)

output = downsample.forward(input)

name = "vae1"
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

save_path = os.path.join(output_dir, "downsample_vae.safetensors")
save_file(tensors, save_path, metadata=metadata)

print(f"\n✓ Successfully generated Downsample2D VAE testcases")
