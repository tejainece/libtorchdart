from typing import List
import torch
from diffusers.models.unets.unet_2d_blocks import DownEncoderBlock2D
import os
from safetensors.torch import save_file
from dataclasses import dataclass

torch.manual_seed(0)
torch_device = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class DownEncoderBlock2DTest:
    name: str
    in_channels: int
    out_channels: int
    num_layers: int
    resnet_eps: float
    resnet_act_fn: str
    resnet_groups: int
    output_scale_factor: float
    add_downsample: bool
    downsample_padding: int

tests: List[DownEncoderBlock2DTest] = [
    DownEncoderBlock2DTest(
        name="simple1",
        in_channels=32,
        out_channels=64,
        num_layers=2,
        resnet_eps=1e-6,
        resnet_act_fn="swish",
        resnet_groups=32,
        output_scale_factor=1.0,
        add_downsample=True,
        downsample_padding=1
    ),
]

tensors = {}
metadata = {}

for test in tests:
    down_block = DownEncoderBlock2D(
        in_channels=test.in_channels,
        out_channels=test.out_channels,
        num_layers=test.num_layers,
        resnet_eps=test.resnet_eps,
        resnet_act_fn=test.resnet_act_fn,
        resnet_groups=test.resnet_groups,
        output_scale_factor=test.output_scale_factor,
        add_downsample=test.add_downsample,
        downsample_padding=test.downsample_padding,
    ).to(torch_device)

    input = torch.randn(1, test.in_channels, 64, 64).to(torch_device)

    output = down_block(input)

    tensors.update({
        f"{test.name}.input": input,
        f"{test.name}.output": output,
        **{f"{test.name}.block.{k}": v for k, v in down_block.state_dict().items()}
    })
    metadata.update({
        f"{test.name}.in_channels": str(test.in_channels),
        f"{test.name}.out_channels": str(test.out_channels),
        f"{test.name}.num_layers": str(test.num_layers),
        f"{test.name}.resnet_eps": str(test.resnet_eps),
        f"{test.name}.resnet_act_fn": str(test.resnet_act_fn),
        f"{test.name}.resnet_groups": str(test.resnet_groups),
        f"{test.name}.output_scale_factor": str(test.output_scale_factor),
        f"{test.name}.add_downsample": str(test.add_downsample),
        f"{test.name}.downsample_padding": str(test.downsample_padding),
    })

os.makedirs("test_data/vae/down_encoder", exist_ok=True)

save_path = f"test_data/vae/down_encoder/downencoder_simple.safetensors"
save_file(tensors, save_path, metadata=metadata)


print("\n✓ Successfully generated DownEncoderBlock2D testcases")
