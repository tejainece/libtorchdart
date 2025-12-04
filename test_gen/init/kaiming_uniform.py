import torch
import math
from torch.nn import functional as F, init

seed=0
size=[32, 32, 3, 3]

torch.manual_seed(seed)

device = 'cpu'

weights = torch.empty(
    size,
    device=device,
)
init.kaiming_uniform_(weights, a=math.sqrt(5))
print(weights)

tesnors = {
    "test1.seed": torch.as_tensor(seed),
    "test1.output": weights,
}

import os
os.makedirs("test_data/init", exist_ok=True)

from safetensors.torch import save_file
save_file(tesnors, "test_data/init/kaiming_uniform_tests.safetensors")