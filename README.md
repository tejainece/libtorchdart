# Roadmap

# Low VRAM

+ Intelligent Tensor offloading
+ Compute a model's and its submodule's memory footprint. It is enough to have one working module in VRAM at a time. See if any single module cannot fit into available VRAM.
+ If VRAM is low, always first load the tensors into RAM, otherwise load them into VRAM directly.

## Phase 0
+ Save Module to disk as Safetensor

## Phase 1: DownSample2D, UpSample2D

## Phase 2: DownEncoderBlock2D, UpDecoderBlock2D

## Phase 4: VAE
+ Test Flux VAE
+ Test SD 1.5 VAE
+ Test SDXL VAE
+ Test Qwen Image VAE

## Phase 5: Unet

## Phase 6: Text encoding

## Phase 7
+ LowVRAM: Comfyui model manager like system to dynamically manage what Tensor is loaded to GPU.
+ Graceful handling of C++ exceptions thrown by libtorch
