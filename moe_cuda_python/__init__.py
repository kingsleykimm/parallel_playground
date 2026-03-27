from . import ops
import torch
from pathlib import Path

so_path = next(Path(__file__).parent.glob("moe_cuda_torch*.so"))
torch.ops.load_library(str(so_path))
