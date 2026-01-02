import kagglehub
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
    return eval(config["model_class"])(
        model_name=config["model_name"], last_layer_dim=config["last_layer_dim"]
    )


def upload_model_to_kaggle():
    LOCAL_MODEL_DIR = "./models"

    MODEL_SLUG = "vit-tiny-patch16-224-splendid-galaxy-22"

    # Learn more about naming model variations at
    # https://www.kaggle.com/docs/models#name-model.
    VARIATION_SLUG = "default"

    kagglehub.model_upload(
        handle=f"matthieuneau/{MODEL_SLUG}/pyTorch/{VARIATION_SLUG}",
        local_model_dir=LOCAL_MODEL_DIR,
        version_notes="Update 2026-01-01",
    )
