import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

import wandb
from image_processing import BiomassDataset
from models import ResNetModel
from utils import enhanced_repr

_standard_repr = torch.Tensor.__repr__

torch.Tensor.__repr__ = enhanced_repr

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

LOSS_WEIGHTS = torch.tensor([0.1, 0.1, 0.1, 0.2, 0.5], device=device)

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

run = wandb.init(
    project="image2biomass",
    config=config,
)

model = ResNetModel().to(device)
criterion = nn.MSELoss(reduction="none")  # Using 'none' to apply custom weights later
optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
transform = transforms.Compose(
    [
        transforms.Lambda(
            lambda img: img.crop((500, 0, 1500, 1000))
        ),  # Crop to center 1000x1000
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)


# Create a generator for deterministic shuffling
rng = torch.Generator().manual_seed(42)

full_dataset = BiomassDataset(
    csv_path="./data/y_train.csv", img_dir="./data/train", transform=transform
)

train_split = 0.8
train_size = int(train_split * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(
    full_dataset, [train_size, val_size], generator=rng
)

train_dataloader = DataLoader(
    train_dataset, batch_size=config["batch_size"], shuffle=True
)
val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)


def train_one_epoch(
    train_dataloader, val_dataloader, model, criterion, optimizer
) -> tuple[float, float, float]:
    # TRAINING
    model.train()
    train_loss, val_loss = 0, 0

    for images, labels in train_dataloader:
        # images: [B, 3, 224, 224], labels: [B, 5]
        images, labels = images.to(device), labels.to(device).float()

        # Forward pass
        preds = model(images)  # Output is [B, 5]
        loss = criterion(preds, labels)
        # Multiply each column by its specific weight
        loss *= LOSS_WEIGHTS
        loss = loss.sum(dim=1).mean()  # Mean over batch

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_dataloader)
    wandb.log({"train_loss": train_loss})

    # VALIDATION
    y_mean = torch.zeros((5,), device=device)
    for images, labels in val_dataloader:
        images, labels = images.to(device), labels.to(device).float()
        y_mean += labels.mean(axis=0)  # TODO: CHECK

        model.eval()
        with torch.no_grad():
            preds = model(images)
            loss = criterion(preds, labels)
            loss *= LOSS_WEIGHTS
            loss = loss.sum(dim=1).mean()  # Mean over batch

        val_loss += loss.item()

    val_loss /= len(val_dataloader)
    y_mean /= len(val_dataloader)

    total_weighted_variance = 0
    for images, labels in val_dataloader:
        images, labels = images.to(device), labels.to(device).float()
        total_weighted_variance += (LOSS_WEIGHTS * (labels - y_mean) ** 2).sum().item()

    R2 = 1 - (val_loss * len(val_dataloader.dataset)) / total_weighted_variance

    wandb.log({"val_loss": val_loss, "val_R2": R2})

    return (train_loss, val_loss, R2)


if __name__ == "__main__":
    n_epochs = config.get("n_epochs", 10)
    for epoch in range(n_epochs):
        train_loss, val_loss, R2 = train_one_epoch(
            train_dataloader, val_dataloader, model, criterion, optimizer
        )
        print(
            f"Epoch [{epoch + 1}/{n_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val R2: {R2:.4f}"
        )
