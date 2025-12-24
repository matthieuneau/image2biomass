import timm
import torch
import torch.nn as nn


class TiledBiomassModel(nn.Module):
    def __init__(self, model_name="resnet18.a1_in1k"):
        super().__init__()
        # 1. Feature Extractor (removes original ImageNet head)
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)

        # 2. Regression Head (processes the 512 features from ResNet18)
        # It predicts the biomass for ONE tile.
        self.regressor = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.ReLU(),  # Add a final ReLU to ensure non-negative output
        )

    def forward(self, x):
        # x shape: [Batch, Tiles, Channels, H, W] -> e.g., [16, 2, 3, 224, 224]
        b, t, c, h, w = x.shape

        # Collapse Batch and Tiles: [32, 3, 224, 224]
        # This makes the model treat every tile as an independent image
        x = x.view(b * t, c, h, w)

        # Extract features and predict per-tile biomass
        features = self.backbone(x)  # Shape: [32, 512]
        tile_preds = self.regressor(features)  # Shape: [32, 1]

        # Reshape back to [Batch, Tiles] and Sum
        tile_preds = tile_preds.view(b, t)  # Shape: [16, 2]
        total_biomass = tile_preds.sum(dim=1)  # Shape: [16] (Sum of both halves)

        return total_biomass


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
model = TiledBiomassModel().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


def train_one_epoch(dataloader, model, criterion, optimizer):
    model.train()
    total_loss = 0

    for images, labels in dataloader:
        # images: [16, 2, 3, 224, 224]
        # labels: [16] (Total biomass for the whole image)
        images, labels = images.to(device), labels.to(device).float()

        # Forward pass
        preds = model(images)  # Output is [16]
        loss = criterion(preds, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)
