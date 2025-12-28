import torch
import torch.nn as nn
import wandb
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms

from image_processing import BiomassDataset
from models import ResNetModel

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

run = wandb.init(
    project="image2biomass",
    config=config,
)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
model = ResNetModel().to(device)
criterion = nn.MSELoss(reduction="none")  # Using 'none' to apply custom weights later
optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
loss_weights = torch.tensor([0.1, 0.1, 0.1, 0.2, 0.5], device=device)

transform = transforms.Compose(
    [
        transforms.Lambda(
            lambda img: img.crop((500, 0, 1500, 1000))
        ),  # Crop to center 1000x1000
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

train_dataset = BiomassDataset(
    csv_path="./data/y_train.csv", img_dir="./data/train", transform=transform
)
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
        # Multiply each column by its specific weight
        loss *= loss_weights
        loss = loss.sum(dim=1).mean()  # Mean over batch

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
