import argparse

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
from image_processing import BiomassDataset
from utils import MODEL_CONFIGS, enhanced_repr, get_model

torch.Tensor.__repr__ = enhanced_repr


def log_dataset_artifact(config):
    """Initializes a brief run to log the dataset artifact once."""
    with wandb.init(
        project="image2biomass", job_type="dataset-upload", config=config
    ) as run:
        artifact = wandb.Artifact(
            name="image2biomass_dataset",
            type="dataset",
            description="Contains all the content of my local ./data folder",
        )
        artifact.add_dir("./data")
        run.log_artifact(artifact)
        print("✅ Dataset artifact logged successfully.")


def train(config=None):
    """The main training function called by the sweep agent or standard run."""
    with wandb.init(project="image2biomass", config=config):
        # Access the resolved config (merges yaml defaults with sweep overrides)
        config = wandb.config

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        # Logic for bfloat16
        amp_dtype = torch.float32
        if config.get("bf16", False) and device.type in ["cuda", "mps"]:
            amp_dtype = torch.bfloat16

        LOSS_WEIGHTS = torch.tensor([0.1, 0.1, 0.1, 0.2, 0.5], device=device)

        # Get model and its configuration
        model = get_model(config).to(device)

        # Get image_size from MODEL_CONFIGS if not in config
        model_name = config["model_name"]
        if "image_size" not in config and model_name in MODEL_CONFIGS:
            image_size = MODEL_CONFIGS[model_name]["image_size"]
        else:
            image_size = config.get("image_size", 224)  # Default to 224

        criterion = nn.MSELoss(reduction="none")
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

        # Data Setup
        full_dataset = BiomassDataset(
            csv_path="./data/y_train.csv",
            img_dir="./data/train",
            transform=model.get_transforms(image_size),
        )
        train_size = int(0.8 * len(full_dataset))
        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, len(full_dataset) - train_size],
            generator=torch.Generator().manual_seed(42),
        )

        train_loader = DataLoader(
            train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4
        )

        best_val_loss = float("inf")
        patience_counter = 0
        n_epochs = config.get("n_epochs", 10)

        for epoch in range(n_epochs):
            model.train()
            train_running_loss = 0.0
            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]"):
                images, labels = images.to(device), labels.to(device).float()

                # Simple implementation - add autocast here if using amp_dtype
                preds = model(images)
                loss = (criterion(preds, labels) * LOSS_WEIGHTS).sum(dim=1).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_running_loss += loss.item()

            # Validation
            model.eval()
            val_running_loss = 0.0
            all_preds, all_labels = [], []
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device).float()
                    preds = model(images)
                    loss = (criterion(preds, labels) * LOSS_WEIGHTS).sum(dim=1).mean()
                    val_running_loss += loss.item()
                    all_preds.append(preds.cpu())
                    all_labels.append(labels.cpu())

            avg_val_loss = val_running_loss / len(val_loader)

            # R2 Calculation
            y_true, y_pred = torch.cat(all_labels), torch.cat(all_preds)
            y_mean = y_true.mean(dim=0)
            ss_res = (LOSS_WEIGHTS.cpu() * (y_true - y_pred) ** 2).sum()
            ss_tot = (LOSS_WEIGHTS.cpu() * (y_true - y_mean) ** 2).sum()
            r2_score = 1 - (ss_res / ss_tot)

            wandb.log(
                {
                    "train_loss": train_running_loss / len(train_loader),
                    "val_loss": avg_val_loss,
                    "val_R2": r2_score.item(),
                }
            )

            # Early Stopping & Model Saving (Tracing for Python 3.14 compatibility)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0

                # Tracing and saving locally
                dummy_input = torch.randn(1, 3, image_size, image_size).to(device)
                traced_model = torch.jit.trace(model, dummy_input)
                model_path = f"models/model_{wandb.run.id}.pt"
                torch.jit.save(traced_model, model_path)
            else:
                patience_counter += 1
                if patience_counter >= config.get("patience", 5):
                    break

        # Log Model Artifact at the end of the run
        model_artifact = wandb.Artifact(
            name=f"{config['model_name']}",
            type="model",
            metadata={"val_loss": best_val_loss, "r2": r2_score.item()},
        )
        model_artifact.add_file(model_path)
        wandb.log_artifact(model_artifact)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true", help="Run in W&B Sweep mode")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    args = parser.parse_args()

    if args.no_wandb:
        import os

        os.environ["WANDB_MODE"] = "disabled"

    with open("config.yaml", "r") as f:
        base_config = yaml.safe_load(f)

    # 1. Log dataset artifact ONCE before training starts (skip if wandb disabled)
    if not args.no_wandb:
        log_dataset_artifact(base_config)

    # 2. Start training or sweep
    if args.sweep:
        with open("sweep.yaml", "r") as f:
            sweep_config = yaml.safe_load(f)

        sweep_id = wandb.sweep(sweep_config, project="image2biomass")
        wandb.agent(sweep_id, function=train)
    else:
        train(config=base_config)
