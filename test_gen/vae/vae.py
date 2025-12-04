import torch
from diffusers import StableDiffusionPipeline, AutoencoderKL
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers import logging
logging.set_verbosity_error()

device = "cuda"
dtype = torch.half

pipe = StableDiffusionPipeline.from_single_file("./models/diffusion/v1-5-pruned-emaonly.safetensors", torch_dtype=torch.float16, use_safetensors=True, local_files_only=True, ).to(device)
#print(pipe.vae.decoder)
print(pipe.vae.quant_conv)

img = Image.open("./images/swordsman1_512.png")
imgTensor = pil_to_tensor(img)[:3, :, :].unsqueeze(0) / 255.0
#print(imgTensor.shape)
#print(imgTensor)
imgTensor = imgTensor.to(device=device, dtype=dtype)

vae: AutoencoderKL = pipe.vae
resp: tuple[DiagonalGaussianDistribution] = vae.encode(imgTensor, return_dict=False)
print(resp)
respDist: DiagonalGaussianDistribution = resp[0]
print(respDist)
print(respDist.mean.shape)
print(respDist.var.shape)
print(respDist.std.shape)
print(respDist.logvar.shape)

print('Finished!')