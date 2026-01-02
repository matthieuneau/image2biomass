import torch

from models import *

_standard_repr = torch.Tensor.__repr__


def enhanced_repr(self) -> str:
    shape = list(self.shape)
    device = self.device
    dtype = str(self.dtype).replace("torch.", "")  # e.g., 'float32'
    grad = "🔥" if self.requires_grad else "❄️"
    mem = f"{self.element_size() * self.nelement() / 1024:.1f}KB"
    return f"[{shape} | {device} | {dtype} | {grad} | {mem}] -> {_standard_repr(self)}"


def get_model(config: dict) -> torch.nn.Module:
    return eval(config["model_class"])(model_name=config["model_name"])
