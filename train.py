import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class ResNetModel(nn.Module):
    def __init__(self, model_name="resnet18.a1_in1k"):
        super().__init__()
        # 1. Feature Extractor (removes original ImageNet head)
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)

        # 2. Regression Head (processes the 512 features from ResNet18)
        self.regressor = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.ReLU(),  # Add a final ReLU to ensure non-negative output
        )

    def forward(self, x):
        # x shape: [B, 3, 224, 224]
        features = self.backbone(x)  # Shape: [B, 512]
        biomass = self.regressor(features)  # Shape: [B, 1]
        return biomass.squeeze(1)  # Shape: [B]


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
model = ResNetModel().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

transform = transforms.Compose(
    [
        transforms.Lambda(
            lambda img: img.crop((500, 0, 1500, 1000))
        ),  # Crop to center 1000x1000
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

train_dataset = datasets.ImageFolder(root="./data/train", transform=transform)
dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)



def train_one_epoch(dataloader, model, criterion, optimizer):
    model.train()
    total_loss = 0

    for images, labels in dataloader:
        # images: [16, 3, 224, 224] (corrected shape)
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


if __name__ == "__main__":
    num_epochs = 1
    for epoch in range(num_epochs):
        avg_loss = train_one_epoch(dataloader, model, criterion, optimizer)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")
