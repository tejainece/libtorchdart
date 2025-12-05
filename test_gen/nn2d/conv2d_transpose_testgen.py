#!/usr/bin/env python3
from typing import List, Literal, LiteralString
import torch
import torch.nn as nn
from safetensors.torch import save_file
import os
from dataclasses import dataclass

device = 'cuda' if torch.cuda.is_available() else 'cpu'

@dataclass
class Conv2DTransposeTestCase:
    name: str
    in_channels: int
    out_channels: int
    groups: int
    kernel_size: tuple | int
    stride: tuple | int
    dilation: tuple | int
    padding: tuple | int
    output_padding: tuple | int
    input_shape: tuple
    has_bias: bool = True

tests: List[Conv2DTransposeTestCase] = [
    Conv2DTransposeTestCase(
        name="basic",
        in_channels=32,
        out_channels=64,
        groups=1,
        kernel_size=3,
        stride=1,
        dilation=1,
        padding=0,
        output_padding=0,
        input_shape=(1, 32, 8, 8),
    ),
    Conv2DTransposeTestCase(
        name="stride2",
        in_channels=32,
        out_channels=16,
        groups=1,
        kernel_size=4,
        stride=2,
        dilation=1,
        padding=1,
        output_padding=0,
        input_shape=(1, 32, 8, 8),
    ),
    Conv2DTransposeTestCase(
        name="output_padding",
        in_channels=16,
        out_channels=16,
        groups=1,
        kernel_size=3,
        stride=2,
        dilation=1,
        padding=1,
        output_padding=1,
        input_shape=(1, 16, 7, 7),
    ),
    Conv2DTransposeTestCase(
        name="non_square_kernel",
        in_channels=8,
        out_channels=16,
        groups=1,
        kernel_size=5,
        stride=1,
        dilation=1,
        padding=0,
        output_padding=0,
        input_shape=(1, 8, 10, 10),
    ),
    Conv2DTransposeTestCase(
        name="asymmetric_stride",
        in_channels=8,
        out_channels=16,
        groups=1,
        kernel_size=3,
        stride=2,
        dilation=1,
        padding=1,
        output_padding=0,
        input_shape=(1, 8, 10, 10),
    ),
    Conv2DTransposeTestCase(
        name="no_bias",
        in_channels=16,
        out_channels=32,
        groups=1,
        kernel_size=3,
        stride=1,
        dilation=1,
        padding=1,
        output_padding=0,
        input_shape=(1, 16, 8, 8),
        has_bias=False,
    ),
    Conv2DTransposeTestCase(
        name="batch_size_4",
        in_channels=16,
        out_channels=32,
        groups=1,
        kernel_size=3,
        stride=2,
        dilation=1,
        padding=1,
        output_padding=0,
        input_shape=(4, 16, 8, 8),
    ),
]

tensors = {}
metadata = {}
torch.manual_seed(42)

for test in tests:    
    conv = nn.ConvTranspose2d(
        in_channels=test.in_channels,
        out_channels=test.out_channels,
        groups=test.groups,
        kernel_size=test.kernel_size,
        stride=test.stride,
        padding=test.padding,
        dilation=test.dilation,
        output_padding=test.output_padding,
        bias=test.has_bias,
    )
    
    input_tensor = torch.randn(test.input_shape)
    output_tensor = conv(input_tensor)
    
    tensors.update({
        f"{test.name}.input": input_tensor,
        f"{test.name}.output": output_tensor,
        **{f"{test.name}.conv.{k}": v for k, v in conv.state_dict().items()},
    })
    
    # Save metadata as a tensor (convert to string representation)
    metadata.update({
        f"{test.name}.in_channels": str(test.in_channels),
        f"{test.name}.out_channels": str(test.out_channels),
        f"{test.name}.groups": str(test.groups),
        f"{test.name}.kernel_size": str(test.kernel_size),
        f"{test.name}.stride": str(test.stride),
        f"{test.name}.dilation": str(test.dilation),
        f"{test.name}.padding": str(test.padding),
        f"{test.name}.output_padding": str(test.output_padding),
        f"{test.name}.has_bias": str(test.has_bias),
    })
    
# Create output directory
output_dir = "test_data/nn2d/conv2d_transpose"
os.makedirs(output_dir, exist_ok=True)

# Save to safetensors
output_path = os.path.join(output_dir, f"simple.safetensors")
save_file(tensors, output_path, metadata=metadata)

print(f"\n✓ Successfully generated Conv2DTranspose simple testcases")
