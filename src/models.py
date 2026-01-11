import timm
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import v2


class CenterCrop:
    """Picklable center crop transform for multiprocessing DataLoader."""
    def __init__(self, left: int = 500, right: int = 1500):
        self.left = left
        self.right = right

    def __call__(self, img):
        return img.crop((self.left, 0, self.right, img.height))


class RandomCrop:
    """Picklable random crop transform that takes a random horizontal slice.

    Similar to CenterCrop but the crop position is randomized.
    Default crop width is 1000px (same as CenterCrop: 1500-500).
    """
    def __init__(self, crop_width: int = 1000):
        self.crop_width = crop_width

    def __call__(self, img):
        max_left = img.width - self.crop_width
        if max_left <= 0:
            return img  # Image is smaller than crop width, return as-is
        left = torch.randint(0, max_left + 1, (1,)).item()
        right = left + self.crop_width
        return img.crop((left, 0, right, img.height))


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
    def get_transforms_static(model_name: str, image_size: int, crop_type: str = "center"):
        """Returns the appropriate transforms based on model architecture.

        Args:
            model_name: Name of the model (for normalization selection)
            image_size: Target image size after resize
            crop_type: "center" for CenterCrop, "random" for RandomCrop
        """
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

        # Select crop transform based on crop_type
        crop_transform = RandomCrop(1000) if crop_type == "random" else CenterCrop(500, 1500)

        return transforms.Compose([
            crop_transform,
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def get_transforms(self, image_size: int, crop_type: str = "center"):
        """Instance method that calls the static method"""
        return UnifiedModel.get_transforms_static(self.model_name, image_size, crop_type)

    @staticmethod
    def get_train_transforms_static(model_name: str, image_size: int, crop_type: str = "center"):
        """Returns training transforms with data augmentation (2025 standard approach).

        Uses torchvision.transforms.v2 with:
        - RandomResizedCrop: Varies scale and aspect ratio
        - RandomHorizontalFlip: Natural for top-down pasture views
        - ColorJitter: Handles varied outdoor lighting conditions
        - TrivialAugmentWide: Automatic augmentation policy (simpler than RandAugment)

        Args:
            model_name: Name of the model (for normalization selection)
            image_size: Target image size after resize
            crop_type: "center" for CenterCrop, "random" for RandomCrop
        """
        # Determine normalization based on model type
        if 'dinov2' in model_name.lower():
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        elif 'vit' in model_name.lower() and 'dinov2' not in model_name.lower():
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
        else:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

        # Select crop transform based on crop_type
        crop_transform = RandomCrop(1000) if crop_type == "random" else CenterCrop(500, 1500)

        return v2.Compose([
            crop_transform,
            v2.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(0.8, 1.0),      # Crop 80-100% of the image
                ratio=(0.9, 1.1),      # Near-square aspect ratio
                antialias=True,
            ),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
            ),
            v2.TrivialAugmentWide(),  # Automatic augmentation policy
            v2.ToImage(),
            v2.ToDtype(dtype=torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        ])

    def get_train_transforms(self, image_size: int, crop_type: str = "center"):
        """Instance method for training transforms with augmentation"""
        return UnifiedModel.get_train_transforms_static(self.model_name, image_size, crop_type)

    def forward(self, x):
        features = self.backbone(x)
        biomass = self.regressor(features)
        return biomass

