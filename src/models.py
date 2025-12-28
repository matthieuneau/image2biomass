import timm
import torch.nn as nn


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

    def forward(self, x):
        # x shape: [B, 3, 224, 224]
        features = self.backbone(x)  # Shape: [B, 512]
        biomass = self.regressor(features)  # Shape: [B, 5]
        return biomass  # Shape: [B, 5]
