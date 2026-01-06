import kagglehub
import torch

from models import UnifiedModel

_standard_repr = torch.Tensor.__repr__

# Model configuration mapping
MODEL_CONFIGS = {
    # ConvNeXt models
    "convnext_tiny.fb_in22k": {"last_layer_dim": 768, "image_size": 224},
    "convnext_nano.d1h_in1k": {"last_layer_dim": 640, "image_size": 224},
    "convnext_pico.d1_in1k": {"last_layer_dim": 512, "image_size": 224},

    # Swin Transformer models
    "swin_tiny_patch4_window7_224": {"last_layer_dim": 768, "image_size": 224},
    "swin_s3_tiny_224": {"last_layer_dim": 768, "image_size": 224},

    # DINOv2 model
    "vit_small_patch14_dinov2.lvd142m": {"last_layer_dim": 384, "image_size": 518},

    # Vanilla ViT models
    "vit_tiny_patch16_224": {"last_layer_dim": 192, "image_size": 224},
    "vit_small_patch16_224": {"last_layer_dim": 384, "image_size": 224},
    "vit_small_patch32_224": {"last_layer_dim": 384, "image_size": 224},

    # ResNet models
    "resnet18": {"last_layer_dim": 512, "image_size": 224},
    "resnet34": {"last_layer_dim": 512, "image_size": 224},
    "resnet50": {"last_layer_dim": 2048, "image_size": 224},

    # EfficientNet models
    "efficientnet_b0": {"last_layer_dim": 1280, "image_size": 224},
    "efficientnet_b1": {"last_layer_dim": 1280, "image_size": 240},
    "efficientnet_b2": {"last_layer_dim": 1408, "image_size": 260},

    # MobileNet model
    "mobilenetv3_large_100": {"last_layer_dim": 1280, "image_size": 224},

    # RegNet models
    "regnetx_016": {"last_layer_dim": 912, "image_size": 224},
    "regnetx_032": {"last_layer_dim": 1008, "image_size": 224},
}


def enhanced_repr(self) -> str:
    shape = list(self.shape)
    device = self.device
    dtype = str(self.dtype).replace("torch.", "")  # e.g., 'float32'
    grad = "🔥" if self.requires_grad else "❄️"
    mem = f"{self.element_size() * self.nelement() / 1024:.1f}KB"
    return f"[{shape} | {device} | {dtype} | {grad} | {mem}] -> {_standard_repr(self)}"


def get_model(config: dict) -> torch.nn.Module:
    """Get model with automatic configuration based on model_name."""
    model_name = config["model_name"]

    # If model has predefined config, use it; otherwise use provided values
    if model_name in MODEL_CONFIGS:
        model_config = MODEL_CONFIGS[model_name]
        # Override config with model-specific values
        config = {**config, **model_config}

    # For backward compatibility, check if last_layer_dim and image_size are provided
    if "last_layer_dim" not in config:
        raise ValueError(f"last_layer_dim not found for model {model_name}")
    if "image_size" not in config:
        raise ValueError(f"image_size not found for model {model_name}")

    return eval(config.get("model_class", "UnifiedModel"))(
        model_name=config["model_name"],
        last_layer_dim=config["last_layer_dim"]
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
