import timm
import torch.nn as nn
from torchvision import transforms


class ResNetModel(nn.Module):
    def __init__(self, model_name="resnet18.a1_in1k"):
        super().__init__()
        # 1. Feature Extractor (removes original ImageNet head)
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)

        # 2. Regression Head (processes the 512 features from ResNet18)
        self.regressor = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 5),  # Predict 5 biomass targets
            nn.ReLU(),  # Add a final ReLU to ensure non-negative output
        )

    @staticmethod
    def get_transforms():
        """Returns the required transforms for ResNet models"""
        return transforms.Compose(
            [
                transforms.Lambda(lambda img: img.crop((500, 0, 1500, 1000))),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def forward(self, x):
        # x shape: [B, 3, 224, 224]
        features = self.backbone(x)  # Shape: [B, 512]
        biomass = self.regressor(features)  # Shape: [B, 5]
        return biomass  # Shape: [B, 5]


class ViTModel(nn.Module):
    def __init__(self, model_name="vit_tiny_patch16_224"):
        super().__init__()
        # 1. Feature Extractor (removes original ImageNet head)
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)

        # 2. Regression Head (processes the 768 features from ViT Tiny)
        self.regressor = nn.Sequential(
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.Linear(128, 5),  # Predict 5 biomass targets
            nn.ReLU(),  # Add a final ReLU to ensure non-negative output
        )

    @staticmethod
    def get_transforms():
        """Returns the required transforms for ViT models"""
        return transforms.Compose(
            [
                transforms.Lambda(lambda img: img.crop((500, 0, 1500, 1000))),
                transforms.Resize((224, 224)),  # TODO: Check impact of resizing
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                ),  # ViT often uses different normalization
            ]
        )

    def forward(self, x):
        # x shape: [B, 3, 224, 224]
        features = self.backbone(x)  # Shape: [B, 768]
        biomass = self.regressor(features)  # Shape: [B, 5]
        return biomass  # Shape: [
