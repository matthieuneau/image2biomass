import argparse
import os
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import wandb
from image_processing import BiomassDataset, TARGET_COLS
from utils import (
    MODEL_CONFIGS,
    compute_per_target_loss,
    compute_per_target_r2,
    compute_residual_stats,
    create_per_target_loss_plot,
    create_per_target_r2_plot,
    create_residual_stats_plot,
    create_scatter_plot,
    enhanced_repr,
    get_gradient_norm,
    get_model,
    log_dataset_artifact,
    log_hard_examples,
)

torch.Tensor.__repr__ = enhanced_repr


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

        # Data Setup - separate transforms for train (with optional augmentation) and val
        use_augmentation = config.get("augment", False)
        crop_type = config.get("crop_type", "center")  # "center" or "random"

        # Get transforms: augmented for training (if enabled), base for validation
        if use_augmentation:
            train_transform = model.get_train_transforms(image_size, crop_type)
        else:
            train_transform = model.get_transforms(image_size, crop_type)
        val_transform = model.get_transforms(image_size, crop_type)

        # Create base dataset to get indices for split
        full_dataset = BiomassDataset(
            csv_path="./data/y_train.csv",
            img_dir="./data/train",
            transform=val_transform,  # Used for hard examples logging
        )
        train_size = int(0.8 * len(full_dataset))

        # Get train/val indices with fixed seed for reproducibility
        generator = torch.Generator().manual_seed(42)
        indices = torch.randperm(len(full_dataset), generator=generator).tolist()
        train_indices = indices[:train_size]
        val_indices_list = indices[train_size:]

        # Create separate datasets with appropriate transforms
        train_dataset_full = BiomassDataset(
            csv_path="./data/y_train.csv",
            img_dir="./data/train",
            transform=train_transform,
        )
        val_dataset_full = BiomassDataset(
            csv_path="./data/y_train.csv",
            img_dir="./data/train",
            transform=val_transform,
        )

        # Use Subset to apply the split indices
        train_dataset = Subset(train_dataset_full, train_indices)
        val_dataset = Subset(val_dataset_full, val_indices_list)

        num_workers = config.get("num_workers", 4)
        prefetch_factor = config.get("prefetch_factor", 2) if num_workers > 0 else None

        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(num_workers > 0),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(num_workers > 0),
        )

        best_val_loss = float("inf")
        patience_counter = 0
        n_epochs = config.get("n_epochs", 10)

        for epoch in range(n_epochs):
            model.train()
            train_running_loss = 0.0
            epoch_start_time = time.time()
            total_samples = 0
            gradient_norms = []

            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]"):
                images, labels = images.to(device), labels.to(device).float()
                total_samples += images.size(0)

                # Simple implementation - add autocast here if using amp_dtype
                preds = model(images)
                loss = (criterion(preds, labels) * LOSS_WEIGHTS).sum(dim=1).mean()

                optimizer.zero_grad()
                loss.backward()

                # Collect gradient norm before optimizer step
                gradient_norms.append(get_gradient_norm(model))

                optimizer.step()
                train_running_loss += loss.item()

            epoch_time = time.time() - epoch_start_time
            samples_per_second = total_samples / epoch_time
            avg_gradient_norm = sum(gradient_norms) / len(gradient_norms)
            max_gradient_norm = max(gradient_norms)

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

            # Compute per-target metrics
            per_target_r2 = compute_per_target_r2(y_true, y_pred, TARGET_COLS)
            per_target_loss = compute_per_target_loss(y_true, y_pred, criterion, TARGET_COLS)
            residual_stats = compute_residual_stats(y_true, y_pred, TARGET_COLS)

            # Build comprehensive log dict
            log_dict = {
                # Original metrics
                "train_loss": train_running_loss / len(train_loader),
                "val_loss": avg_val_loss,
                "val_R2": r2_score.item(),
                # Throughput metrics
                "epoch_time_seconds": epoch_time,
                "samples_per_second": samples_per_second,
                # Gradient norms
                "gradient_norm_avg": avg_gradient_norm,
                "gradient_norm_max": max_gradient_norm,
                # Per-target metrics
                **per_target_r2,
                **per_target_loss,
                **residual_stats,
            }

            # Log combined plots every 5 epochs + final epoch
            is_final_epoch = (epoch == n_epochs - 1) or (
                patience_counter >= config.get("patience", 5) - 1
            )
            if (epoch + 1) % 5 == 0 or is_final_epoch:
                y_true_np = y_true.numpy()
                y_pred_np = y_pred.numpy()

                # Scatter plots for each target
                for i, name in enumerate(TARGET_COLS):
                    fig = create_scatter_plot(y_true_np[:, i], y_pred_np[:, i], name)
                    log_dict[f"scatter_{name}"] = wandb.Image(fig)
                    plt.close(fig)

                # Combined per-target loss plot
                fig_loss = create_per_target_loss_plot(per_target_loss, TARGET_COLS)
                log_dict["per_target_loss_plot"] = wandb.Image(fig_loss)
                plt.close(fig_loss)

                # Combined per-target R2 plot
                fig_r2 = create_per_target_r2_plot(per_target_r2, TARGET_COLS)
                log_dict["per_target_r2_plot"] = wandb.Image(fig_r2)
                plt.close(fig_r2)

                # Combined residual stats plot
                fig_residuals = create_residual_stats_plot(residual_stats, TARGET_COLS)
                log_dict["residual_stats_plot"] = wandb.Image(fig_residuals)
                plt.close(fig_residuals)

            wandb.log(log_dict)

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

        # Log hard examples only once at the end of training
        hard_examples_table = log_hard_examples(
            full_dataset, val_indices_list, y_true, y_pred, TARGET_COLS, n_examples=10
        )
        wandb.log({"hard_examples": hard_examples_table})

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
