"""
Per-class head experiment: Train one distinct regression head per target.

This script loads a pretrained backbone from wandb and trains 5 separate
heads, each specialized for predicting one biomass target.
"""

import argparse
import os

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import wandb
from image_processing import TARGET_COLS, BiomassDataset
from models import Regressor, UnifiedModel
from utils import MODEL_CONFIGS

# =============================================================================
# CONFIGURATION
# =============================================================================

# Backbone to load from wandb (entity/project/artifact:version)
BACKBONE_ARTIFACT = "matthieu-neau/image2biomass/vit_tiny_patch16_224:v52"

# Model settings (must match the backbone)
MODEL_NAME = "vit_tiny_patch16_224"

# Training settings
CONFIG = {
    "lr": 0.0001,
    "n_epochs": 50,
    "batch_size": 16,
    "patience": 5,
    "head_hidden_dim": 64,
    "head_num_layers": 1,
    "head_dropout": 0.2,
    "head_activation": "relu",
    "head_output_activation": "relu",
    "head_normalization": "none",
    "optimizer": "adam",
    "weight_decay": 0.01,
    "train_split": 0.8,
}

LOSS_WEIGHTS = [0.1, 0.1, 0.1, 0.2, 0.5]  # Per-target weights for final metric

# =============================================================================
# MODEL
# =============================================================================


class PerClassHeadModel(nn.Module):
    """Model with frozen backbone and separate heads for each target."""

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        head_hidden_dim: int = 64,
        head_num_layers: int = 1,
        head_dropout: float = 0.2,
        head_activation: str = "relu",
        head_output_activation: str = "relu",
        head_normalization: str = "none",
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim

        # Freeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Create one head per target
        self.heads = nn.ModuleList(
            [
                Regressor(
                    in_dim=feature_dim,
                    hidden_dim=head_hidden_dim,
                    num_layers=head_num_layers,
                    dropout=head_dropout,
                    activation=head_activation,
                    output_activation=head_output_activation,
                    normalization=head_normalization,
                    num_outputs=1,  # Each head predicts 1 target
                )
                for _ in range(len(TARGET_COLS))
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: extract features, predict each target separately."""
        with torch.no_grad():
            features = self.backbone(x)

        # Each head predicts its target
        predictions = [head(features) for head in self.heads]
        return torch.cat(predictions, dim=1)  # Shape: (batch, 5)

    def forward_single(self, x: torch.Tensor, target_idx: int) -> torch.Tensor:
        """Forward pass for a single target (for per-head training)."""
        with torch.no_grad():
            features = self.backbone(x)
        return self.heads[target_idx](features)


# =============================================================================
# TRAINING
# =============================================================================


def train_per_class_heads(config: dict):
    """Train separate heads for each target class."""

    with wandb.init(project="image2biomass", config=config, tags=["per-class-head"]):
        config = wandb.config
        wandb.run.name = f"per-class-{MODEL_NAME}"

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"Using device: {device}")

        # Get model config
        model_config = MODEL_CONFIGS.get(MODEL_NAME, {})
        feature_dim = model_config.get("last_layer_dim", 192)
        image_size = model_config.get("image_size", 224)

        # Download backbone from wandb
        print(f"Downloading backbone from {BACKBONE_ARTIFACT}...")
        artifact = wandb.use_artifact(BACKBONE_ARTIFACT, type="model")
        artifact_dir = artifact.download()

        # Find backbone file
        backbone_files = [
            f for f in os.listdir(artifact_dir) if f.startswith("backbone_")
        ]
        if not backbone_files:
            # Fall back to full model and extract backbone
            model_files = [
                f for f in os.listdir(artifact_dir) if f.startswith("model_")
            ]
            if not model_files:
                raise FileNotFoundError(f"No model files found in {artifact_dir}")
            model_path = os.path.join(artifact_dir, model_files[0])
            print(f"Loading full model from {model_path} (will use as backbone)")
            # Load traced model - we'll need to extract features differently
            full_model = torch.jit.load(model_path, map_location=device)
            # For traced models, we can't easily extract backbone, so we'll use it as-is
            # and add heads on top (predictions will be ignored)
            raise NotImplementedError(
                "Backbone file not found. Please ensure the artifact contains backbone_*.pt. "
                "Re-run training with the updated code that saves backbones separately."
            )
        else:
            backbone_path = os.path.join(artifact_dir, backbone_files[0])
            print(f"Loading backbone from {backbone_path}")
            backbone = torch.jit.load(backbone_path, map_location=device)

        # Create model with per-class heads
        model = PerClassHeadModel(
            backbone=backbone,
            feature_dim=feature_dim,
            head_hidden_dim=config.get("head_hidden_dim", 64),
            head_num_layers=config.get("head_num_layers", 1),
            head_dropout=config.get("head_dropout", 0.2),
            head_activation=config.get("head_activation", "relu"),
            head_output_activation=config.get("head_output_activation", "relu"),
            head_normalization=config.get("head_normalization", "none"),
            freeze_backbone=True,
        ).to(device)

        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
        wandb.log({"trainable_params": trainable_params, "total_params": total_params})

        # Setup data
        transform = UnifiedModel.get_transforms_static(MODEL_NAME, image_size)
        full_dataset = BiomassDataset(
            csv_path="./data/y_train.csv",
            img_dir="./data/train",
            transform=transform,
        )

        # Train/val split
        generator = torch.Generator().manual_seed(42)
        indices = torch.randperm(len(full_dataset), generator=generator).tolist()
        train_size = int(config.get("train_split", 0.8) * len(full_dataset))
        train_indices, val_indices = indices[:train_size], indices[train_size:]

        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)

        train_loader = DataLoader(
            train_dataset, batch_size=config.get("batch_size", 16), shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config.get("batch_size", 16), shuffle=False
        )

        # Setup optimizer (only for heads)
        if config.get("optimizer", "adam") == "adamw":
            optimizer = torch.optim.AdamW(
                model.heads.parameters(),
                lr=config.get("lr", 0.0001),
                weight_decay=config.get("weight_decay", 0.01),
            )
        else:
            optimizer = torch.optim.Adam(
                model.heads.parameters(),
                lr=config.get("lr", 0.0001),
                weight_decay=config.get("weight_decay", 0.01),
            )

        criterion = nn.MSELoss(reduction="none")
        loss_weights = torch.tensor(LOSS_WEIGHTS, device=device)

        best_val_loss = float("inf")
        patience_counter = 0

        print(f"\nTraining {len(TARGET_COLS)} separate heads...")
        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

        for epoch in range(config.get("n_epochs", 50)):
            # Training
            model.train()
            train_losses = []

            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1} Train"):
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                preds = model(images)

                # Weighted MSE loss
                per_target_loss = criterion(preds, labels)
                loss = (per_target_loss * loss_weights).sum(dim=1).mean()

                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            avg_train_loss = sum(train_losses) / len(train_losses)

            # Validation
            model.eval()
            val_losses = []
            all_preds, all_labels = [], []

            with torch.no_grad():
                for images, labels in tqdm(val_loader, desc=f"Epoch {epoch + 1} Val"):
                    images, labels = images.to(device), labels.to(device)
                    preds = model(images)

                    per_target_loss = criterion(preds, labels)
                    loss = (per_target_loss * loss_weights).sum(dim=1).mean()
                    val_losses.append(loss.item())

                    all_preds.append(preds.cpu())
                    all_labels.append(labels.cpu())

            avg_val_loss = sum(val_losses) / len(val_losses)
            all_preds = torch.cat(all_preds, dim=0)
            all_labels = torch.cat(all_labels, dim=0)

            # Compute R² per target
            per_target_r2 = {}
            for i, name in enumerate(TARGET_COLS):
                ss_res = ((all_labels[:, i] - all_preds[:, i]) ** 2).sum()
                ss_tot = ((all_labels[:, i] - all_labels[:, i].mean()) ** 2).sum()
                r2 = 1 - (ss_res / ss_tot)
                per_target_r2[name] = r2.item()

            # Weighted R²
            ss_res_weighted = (loss_weights.cpu() * (all_labels - all_preds) ** 2).sum()
            ss_tot_weighted = (
                loss_weights.cpu() * (all_labels - all_labels.mean(dim=0)) ** 2
            ).sum()
            weighted_r2 = 1 - (ss_res_weighted / ss_tot_weighted)

            # Log metrics
            log_dict = {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_R2": weighted_r2.item(),
            }
            for name, r2 in per_target_r2.items():
                log_dict[f"val_R2_{name}"] = r2

            wandb.log(log_dict)

            print(
                f"Epoch {epoch + 1}: Train={avg_train_loss:.4f}, "
                f"Val={avg_val_loss:.4f}, R²={weighted_r2:.4f}"
            )
            for name, r2 in per_target_r2.items():
                print(f"  {name}: R²={r2:.4f}")

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0

                # Save model
                model_path = f"models/per_class_head_{wandb.run.id}.pt"
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": dict(config),
                        "model_name": MODEL_NAME,
                        "feature_dim": feature_dim,
                    },
                    model_path,
                )
                print(f"  Saved best model to {model_path}")
            else:
                patience_counter += 1
                if patience_counter >= config.get("patience", 10):
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        # Final summary
        print(f"\nBest validation loss: {best_val_loss:.4f}")

        # Create per-target R² bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        names = list(per_target_r2.keys())
        values = list(per_target_r2.values())
        colors = ["green" if v > 0 else "red" for v in values]
        ax.bar(names, values, color=colors)
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax.set_ylabel("R²")
        ax.set_title("Per-Target R² (Per-Class Heads)")
        ax.set_ylim(-1, 1)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        wandb.log({"per_target_r2_chart": wandb.Image(fig)})
        plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, help="Wandb artifact path for backbone")
    parser.add_argument("--model-name", type=str, help="Model name (for config lookup)")
    args = parser.parse_args()

    if args.backbone:
        BACKBONE_ARTIFACT = args.backbone
    if args.model_name:
        MODEL_NAME = args.model_name

    train_per_class_heads(CONFIG)
