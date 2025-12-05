from typing import Literal
import torch

from dataclasses import dataclass

torch.manual_seed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

@dataclass
class Conv2DParams:
    name: str
    in_channels: int
    out_channels: int
    kernel_size: int | tuple[int, int]
    padding: int | tuple[int, int]
    stride: int | tuple[int, int]
    dilation: int | tuple[int, int]
    groups: int
    padding_mode: Literal["zeros", "reflect", "replicate", "circular"]

tests: list[Conv2DParams] = [
    Conv2DParams(
        name="simple",
        in_channels=32,
        out_channels=32,
        kernel_size=3,
        padding=0,
        stride=1,
        dilation=1,
        groups=1,
        padding_mode="zeros",
    ),
    Conv2DParams(
        name="dilation",
        in_channels=32,
        out_channels=32,
        kernel_size=3,
        padding=0,
        stride=1,
        dilation=2,
        groups=1,
        padding_mode="zeros",
    ),
    Conv2DParams(
        name="zeros",
        in_channels=32,
        out_channels=32,
        kernel_size=3,
        padding=1,
        stride=1,
        dilation=1,
        groups=1,
        padding_mode="zeros",
    ),
    Conv2DParams(
        name="reflect",
        in_channels=32,
        out_channels=32,
        kernel_size=3,
        padding=1,
        stride=1,
        dilation=1,
        groups=1,
        padding_mode="reflect",
    ),
    Conv2DParams(
        name="replicate",
        in_channels=32,
        out_channels=32,
        kernel_size=3,
        padding=1,
        stride=1,
        dilation=1,
        groups=1,
        padding_mode="replicate",
    ),
    Conv2DParams(
        name="circular",
        in_channels=32,
        out_channels=32,
        kernel_size=3,
        padding=1,
        stride=1,
        dilation=1,
        groups=1,
        padding_mode="circular",
    ),
]

tensors = {}
metadata = {}

for test in tests:
    conv = torch.nn.Conv2d(
        in_channels=test.in_channels, 
        out_channels=test.out_channels, 
        kernel_size=test.kernel_size, 
        padding=test.padding, 
        stride=test.stride,
        dilation=test.dilation,
        groups=test.groups, 
        padding_mode=test.padding_mode, 
        device=device)

    #print(conv)
    #print(conv.weight)
    #print(conv.bias)
    input = torch.ones([1, 32, 28, 28], device=device)
    output = conv.forward(input)
    #print(out.shape)
    #print(out)

    tensors.update({
        f"{test.name}.input": input,
        f"{test.name}.output": output,
        **{f"{test.name}.conv.{k}": v for k, v in conv.state_dict().items()},
    })
    metadata.update({
        f"{test.name}.padding": str(conv.padding),
        f"{test.name}.stride": str(conv.stride),
        f"{test.name}.dilation": str(conv.dilation),
        f"{test.name}.groups": str(conv.groups),
        f"{test.name}.padding_mode": str(conv.padding_mode),
    })
    print(conv.padding_mode)

import os
os.makedirs("test_data/nn2d/conv2d", exist_ok=True)

from safetensors.torch import save_file
save_file(tensors, f"test_data/nn2d/conv2d/conv2d_simple.safetensors", metadata=metadata)

print(f"\n✓ Successfully generated Conv2D simple testcases")

