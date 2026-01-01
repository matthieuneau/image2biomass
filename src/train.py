import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
from image_processing import BiomassDataset
from models import ViTModel
from utils import enhanced_repr

# ... [Device and Repr setup remains the same] ...
torch.Tensor.__repr__ = enhanced_repr

# Load Config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Initialization
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
if config.get("bf16", False) and device.type in ["cuda", "mps"]:
    amp_dtype = torch.bfloat16

artifact = wandb.Artifact(
    name="image2biomass_dataset",
    type="dataset",
    description="Contains all the content of my local ./data folder",
)
artifact.add_dir("./data")
run.log_artifact(artifact)

LOSS_WEIGHTS = torch.tensor([0.1, 0.1, 0.1, 0.2, 0.5], device=device)
model = ViTModel().to(device)
criterion = nn.MSELoss(reduction="none")
optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

# Data Setup
full_dataset = BiomassDataset(
    csv_path="./data/y_train.csv",
    img_dir="./data/train",
    transform=model.get_transforms(),
)
train_size = int(0.8 * len(full_dataset))
train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, len(full_dataset) - train_size],
    generator=torch.Generator().manual_seed(42),
)

train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

wandb.init(project="image2biomass", config=config)

# --- MAIN TRAINING LOOP ---
n_epochs = config.get("n_epochs", 10)

for epoch in range(n_epochs):
    # 1. TRAINING PHASE
    model.train()
    train_running_loss = 0.0

    # Progress bar for training
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{n_epochs} [Train]")
    for images, labels in train_bar:
        images, labels = images.to(device), labels.to(device).float()

        preds = model(images)
        loss = (criterion(preds, labels) * LOSS_WEIGHTS).sum(dim=1).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_running_loss += loss.item()
        train_bar.set_postfix(loss=loss.item())

    avg_train_loss = train_running_loss / len(train_loader)

    # 2. VALIDATION PHASE
    model.eval()
    val_running_loss = 0.0
    all_preds = []
    all_labels = []

    val_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{n_epochs} [Val]")
    with torch.no_grad():
        for images, labels in val_bar:
            images, labels = images.to(device), labels.to(device).float()

            preds = model(images)
            loss = (criterion(preds, labels) * LOSS_WEIGHTS).sum(dim=1).mean()

            val_running_loss += loss.item()

            # Store for R2 calculation (keep on CPU to save MPS/GPU memory)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            val_bar.set_postfix(loss=loss.item())

    # 3. METRICS CALCULATION
    avg_val_loss = val_running_loss / len(val_loader)

    # Concatenate all batches for a global R2
    y_true = torch.cat(all_labels)
    y_pred = torch.cat(all_preds)

    # Weighted R2 Logic
    y_mean = y_true.mean(dim=0)
    ss_res = (LOSS_WEIGHTS.cpu() * (y_true - y_pred) ** 2).sum()
    ss_tot = (LOSS_WEIGHTS.cpu() * (y_true - y_mean) ** 2).sum()
    r2_score = 1 - (ss_res / ss_tot)

    # 4. LOGGING
    print(
        f"\n✨ Epoch {epoch + 1} Summary: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | R2: {r2_score:.4f}\n"
    )
    wandb.log(
        {
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_R2": r2_score.item(),
        }
    )

wandb.finish()
