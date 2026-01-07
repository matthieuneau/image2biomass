import timm
import torch.nn as nn
from torchvision import transforms


class CenterCrop:
    """Picklable center crop transform for multiprocessing DataLoader."""
    def __init__(self, left: int = 500, right: int = 1500):
        self.left = left
        self.right = right

    def __call__(self, img):
        return img.crop((self.left, 0, self.right, img.height))


class UnifiedModel(nn.Module):
    """Unified model class that can work with any timm backbone"""

    def __init__(self, model_name: str, last_layer_dim: int, pretrained: bool = True):
        super().__init__()
        self.model_name = model_name

        # 1. Feature Extractor (removes original head)
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)

        # 2. Regression Head
        self.regressor = nn.Sequential(
            nn.Linear(last_layer_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 5),  # Predict 5 biomass targets
            nn.ReLU(),  # Ensure non-negative output
        )

    @staticmethod
    def get_transforms_static(model_name: str, image_size: int):
        """Returns the appropriate transforms based on model architecture"""

        # Determine normalization based on model type
        if 'dinov2' in model_name.lower():
            # DINOv2 models use ImageNet normalization
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        elif 'vit' in model_name.lower() and 'dinov2' not in model_name.lower():
            # Standard ViT models often use different normalization
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
        else:
            # ConvNets (ResNet, EfficientNet, ConvNeXt, etc.) use ImageNet stats
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

        return transforms.Compose([
            CenterCrop(500, 1500),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def get_transforms(self, image_size: int):
        """Instance method that calls the static method"""
        return UnifiedModel.get_transforms_static(self.model_name, image_size)

    def forward(self, x):
        features = self.backbone(x)
        biomass = self.regressor(features)
        return biomass

