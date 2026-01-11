import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.model_selection import KFold, LeaveOneGroupOut
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import wandb
from image_processing import TARGET_COLS, BiomassDataset
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
    log_crop_variance_analysis,
    log_dataset_artifact,
    log_hard_examples,
)

torch.Tensor.__repr__ = enhanced_repr
LOSS_WEIGHTS_LIST = [0.1, 0.1, 0.1, 0.2, 0.5]


def get_cv_splits(n_samples: int, config: dict, groups: list | None = None):
    """Generate train/val splits based on CV config."""
    cv_type = config.get("cv_type", "kfold")
    cv_folds = config.get("cv_folds", 5)
    indices = np.arange(n_samples)

    if cv_type == "region":
        if groups is None:
            raise ValueError("Region-wise CV requires group labels (states)")
        logo = LeaveOneGroupOut()
        for fold_idx, (train_idx, val_idx) in enumerate(
            logo.split(indices, groups=groups)
        ):
            val_group = groups[val_idx[0]]
            yield (
                fold_idx,
                train_idx.tolist(),
                val_idx.tolist(),
                f"fold_{fold_idx + 1}_val_{val_group}",
            )
    else:
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(indices)):
            yield (
                fold_idx,
                train_idx.tolist(),
                val_idx.tolist(),
                f"fold_{fold_idx + 1}_of_{cv_folds}",
            )


def create_dataloaders(train_dataset, val_dataset, config, device):
    """Create train and val dataloaders with common settings."""
    num_workers = config.get("num_workers", 4)
    prefetch = config.get("prefetch_factor", 2) if num_workers > 0 else None
    common_kwargs = {
        "batch_size": config["batch_size"],
        "num_workers": num_workers,
        "prefetch_factor": prefetch,
        "pin_memory": (device.type == "cuda"),
        "persistent_workers": (num_workers > 0),
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **common_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **common_kwargs)
    return train_loader, val_loader


def train_one_fold(
    config,
    device,
    train_indices,
    val_indices,
    train_dataset_full,
    val_dataset_full,
    full_dataset,
    fold_name="",
    log_plots=True,
):
    """Train a single fold and return results."""
    LOSS_WEIGHTS = torch.tensor(LOSS_WEIGHTS_LIST, device=device)
    prefix = f"{fold_name}/" if fold_name else ""

    # Setup model and training
    model = get_model(config).to(device)
    model_name = config["model_name"]
    image_size = MODEL_CONFIGS.get(model_name, {}).get(
        "image_size", config.get("image_size", 224)
    )
    criterion = nn.MSELoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    # Create dataloaders
    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(val_dataset_full, val_indices)
    train_loader, val_loader = create_dataloaders(
        train_dataset, val_dataset, config, device
    )

    best_val_loss, patience_counter = float("inf"), 0
    n_epochs = config.get("n_epochs", 10)
    best_state, model_path = None, None

    print(
        f"\n{'=' * 60}\nTraining {fold_name or 'model'} | Train: {len(train_indices)}, Val: {len(val_indices)}\n{'=' * 60}"
    )

    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss, total_samples, grad_norms = 0.0, 0, []
        epoch_start = time.time()

        for images, labels in tqdm(
            train_loader, desc=f"{prefix}Epoch {epoch + 1} [Train]"
        ):
            images, labels = images.to(device), labels.to(device).float()
            total_samples += images.size(0)

            preds = model(images)
            loss = (criterion(preds, labels) * LOSS_WEIGHTS).sum(dim=1).mean()

            optimizer.zero_grad()
            loss.backward()
            grad_norms.append(get_gradient_norm(model))
            optimizer.step()
            train_loss += loss.item()

        epoch_time = time.time() - epoch_start

        # Validation
        model.eval()
        val_loss, all_preds, all_labels = 0.0, [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).float()
                preds = model(images)
                loss = (criterion(preds, labels) * LOSS_WEIGHTS).sum(dim=1).mean()
                val_loss += loss.item()
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        avg_val_loss = val_loss / len(val_loader)
        y_true, y_pred = torch.cat(all_labels), torch.cat(all_preds)

        # Compute R2
        ss_res = (LOSS_WEIGHTS.cpu() * (y_true - y_pred) ** 2).sum()
        ss_tot = (LOSS_WEIGHTS.cpu() * (y_true - y_true.mean(dim=0)) ** 2).sum()
        r2_score = 1 - (ss_res / ss_tot)

        # Per-target metrics
        per_target_r2 = compute_per_target_r2(y_true, y_pred, TARGET_COLS)
        per_target_loss = compute_per_target_loss(
            y_true, y_pred, criterion, TARGET_COLS
        )
        residual_stats = compute_residual_stats(y_true, y_pred, TARGET_COLS)

        # Log metrics
        log_dict = {
            f"{prefix}train_loss": train_loss / len(train_loader),
            f"{prefix}val_loss": avg_val_loss,
            f"{prefix}val_R2": r2_score.item(),
            f"{prefix}epoch_time_seconds": epoch_time,
            f"{prefix}samples_per_second": total_samples / epoch_time,
            f"{prefix}gradient_norm_avg": sum(grad_norms) / len(grad_norms),
            f"{prefix}gradient_norm_max": max(grad_norms),
            **{
                f"{prefix}{k}": v
                for k, v in {
                    **per_target_r2,
                    **per_target_loss,
                    **residual_stats,
                }.items()
            },
        }

        # Log plots periodically
        is_final = (epoch == n_epochs - 1) or (
            patience_counter >= config.get("patience", 5) - 1
        )
        if log_plots and ((epoch + 1) % 5 == 0 or is_final):
            y_true_np, y_pred_np = y_true.numpy(), y_pred.numpy()
            for i, name in enumerate(TARGET_COLS):
                fig = create_scatter_plot(y_true_np[:, i], y_pred_np[:, i], name)
                log_dict[f"{prefix}scatter_{name}"] = wandb.Image(fig)
                plt.close(fig)
            for create_fn, key, data in [
                (create_per_target_loss_plot, "per_target_loss_plot", per_target_loss),
                (create_per_target_r2_plot, "per_target_r2_plot", per_target_r2),
                (create_residual_stats_plot, "residual_stats_plot", residual_stats),
            ]:
                fig = create_fn(data, TARGET_COLS)
                log_dict[f"{prefix}{key}"] = wandb.Image(fig)
                plt.close(fig)

        wandb.log(log_dict)

        # Early stopping & model saving
        if avg_val_loss < best_val_loss:
            best_val_loss, patience_counter = avg_val_loss, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            dummy = torch.randn(1, 3, image_size, image_size).to(device)
            traced = torch.jit.trace(model, dummy)
            model_path = (
                f"models/model_{wandb.run.id}{'_' + fold_name if fold_name else ''}.pt"
            )
            torch.jit.save(traced, model_path)
        else:
            patience_counter += 1
            if patience_counter >= config.get("patience", 5):
                print(f"Early stopping at epoch {epoch + 1}")
                break

    return {
        "fold_name": fold_name,
        "best_val_loss": best_val_loss,
        "final_r2": r2_score.item(),
        "per_target_r2": per_target_r2,
        "y_true": y_true,
        "y_pred": y_pred,
        "val_indices": val_indices,
        "model_state": best_state,
        "model_path": model_path,
        "image_size": image_size,
    }


def train(config=None):
    """Main training function for single run or CV."""
    with wandb.init(project="image2biomass", config=config):
        config = wandb.config
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        # Setup transforms
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
        crop_width = config.get("crop_width", 1000)  # Width of crop in pixels

        # Get transforms: augmented for training (if enabled), base for validation
        if use_augmentation:
            train_transform = model.get_train_transforms(
                image_size, crop_type, crop_width
            )
        else:
            train_transform = model.get_transforms(image_size, crop_type, crop_width)
        val_transform = model.get_transforms(image_size, crop_type, crop_width)

        # Create base dataset to get indices for split
        full_dataset = BiomassDataset(
            csv_path="./data/y_train.csv",
            img_dir="./data/train",
            transform=val_transform,  # Used for hard examples logging
        )
        temp_model = get_model(config)
        train_transform = (
            temp_model.get_train_transforms(image_size)
            if config.get("augment", False)
            else temp_model.get_transforms(image_size)
        )
        val_transform = temp_model.get_transforms(image_size)
        del temp_model

        # Create datasets
        cv_type = config.get("cv_type", "kfold")
        metadata_csv = "./data/train.csv" if cv_type == "region" else None
        dataset_kwargs = {
            "csv_path": "./data/y_train.csv",
            "img_dir": "./data/train",
            "metadata_csv": metadata_csv,
        }

        full_dataset = BiomassDataset(**dataset_kwargs, transform=val_transform)
        train_dataset_full = BiomassDataset(**dataset_kwargs, transform=train_transform)
        val_dataset_full = BiomassDataset(**dataset_kwargs, transform=val_transform)

        cv_enabled = config.get("cv_enabled", False)

        if cv_enabled:
            # Cross-validation mode
            print(f"\n{'#' * 60}\nCross-Validation Mode: {cv_type}\n{'#' * 60}")
            groups = full_dataset.get_states() if cv_type == "region" else None
            fold_results = []

            for fold_idx, train_idx, val_idx, fold_name in get_cv_splits(
                len(full_dataset), config, groups
            ):
                result = train_one_fold(
                    config,
                    device,
                    train_idx,
                    val_idx,
                    train_dataset_full,
                    val_dataset_full,
                    full_dataset,
                    fold_name=fold_name,
                    log_plots=False,
                )
                fold_results.append(result)

            # Aggregate and log CV summary
            avg_loss = np.mean([r["best_val_loss"] for r in fold_results])
            std_loss = np.std([r["best_val_loss"] for r in fold_results])
            avg_r2 = np.mean([r["final_r2"] for r in fold_results])
            std_r2 = np.std([r["final_r2"] for r in fold_results])

            cv_summary = {
                "cv_avg_val_loss": avg_loss,
                "cv_std_val_loss": std_loss,
                "cv_avg_R2": avg_r2,
                "cv_std_R2": std_r2,
            }
            for key in fold_results[0]["per_target_r2"]:
                vals = [r["per_target_r2"][key] for r in fold_results]
                cv_summary[f"cv_avg_{key}"] = np.mean(vals)
                cv_summary[f"cv_std_{key}"] = np.std(vals)
            wandb.log(cv_summary)

            wandb.log(
                {
                    "cv_fold_summary": wandb.Table(
                        columns=["fold", "val_loss", "R2"],
                        data=[
                            [r["fold_name"], r["best_val_loss"], r["final_r2"]]
                            for r in fold_results
                        ],
                    )
                }
            )

            print(f"\n{'=' * 60}")
            print(
                f"CV Summary: Val Loss = {avg_loss:.4f} +/- {std_loss:.4f}, R2 = {avg_r2:.4f} +/- {std_r2:.4f}"
            )
            print(f"{'=' * 60}")

            # Save best fold model as artifact
            best = min(fold_results, key=lambda x: x["best_val_loss"])
            artifact = wandb.Artifact(
                name=f"{config['model_name']}_cv",
                type="model",
                metadata={
                    "cv_type": cv_type,
                    "cv_avg_val_loss": avg_loss,
                    "cv_avg_r2": avg_r2,
                    "best_fold": best["fold_name"],
                },
            )
            artifact.add_file(best["model_path"])
            wandb.log_artifact(artifact)

        else:
            # Standard single split training
            generator = torch.Generator().manual_seed(42)
            indices = torch.randperm(len(full_dataset), generator=generator).tolist()
            train_size = int(0.8 * len(full_dataset))
            train_indices, val_indices = indices[:train_size], indices[train_size:]

            result = train_one_fold(
                config,
                device,
                train_indices,
                val_indices,
                train_dataset_full,
                val_dataset_full,
                full_dataset,
            )

            # TTA evaluation every 5 epochs after epoch 30
            tta_enabled = config.get("tta", False)
            if tta_enabled and (epoch + 1) >= 30 and (epoch + 1) % 5 == 0:
                from PIL import Image

                tta_tiles = config.get("tta_tiles", 18)
                tile_size = config.get("tile_size", 500)

                model.eval()
                tta_preds, tta_labels = [], []

                with torch.no_grad():
                    for idx in tqdm(
                        val_indices_list, desc=f"TTA Eval (epoch {epoch + 1})"
                    ):
                        img_name = full_dataset.df.iloc[idx]["image_id"]
                        img_path = f"./data/train/{img_name}.jpg"
                        raw_img = Image.open(img_path).convert("RGB")

                        pred = model.predict_with_tta(
                            raw_image=raw_img,
                            image_size=image_size,
                            tile_size=tile_size,
                            tta_tiles=tta_tiles,
                            device=device,
                        )
                        tta_preds.append(pred.unsqueeze(0))

                        label = torch.tensor(
                            full_dataset.df.iloc[idx][TARGET_COLS].values.astype(float)
                        )
                        tta_labels.append(label.unsqueeze(0))

                tta_preds = torch.cat(tta_preds, dim=0).cpu()
                tta_labels = torch.cat(tta_labels, dim=0).cpu()

                # Compute TTA metrics
                tta_loss = (
                    (criterion(tta_preds, tta_labels) * LOSS_WEIGHTS.cpu())
                    .sum(dim=1)
                    .mean()
                )
                tta_ss_res = (LOSS_WEIGHTS.cpu() * (tta_labels - tta_preds) ** 2).sum()
                tta_ss_tot = (
                    LOSS_WEIGHTS.cpu() * (tta_labels - tta_labels.mean(dim=0)) ** 2
                ).sum()
                tta_r2 = 1 - (tta_ss_res / tta_ss_tot)

                log_dict["tta_val_loss"] = tta_loss.item()
                log_dict["tta_val_R2"] = tta_r2.item()

                print(
                    f"  TTA (tiles={tta_tiles}): Loss={tta_loss.item():.4f}, R2={tta_r2.item():.4f}"
                )

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

        # Log crop variance analysis (only meaningful for random crops)
        if crop_type == "random":
            crop_variance_table = log_crop_variance_analysis(
                model=model,
                dataset=full_dataset,
                device=device,
                image_size=image_size,
                crop_width=crop_width,
                target_cols=TARGET_COLS,
                n_images=10,
                n_crops=30,
            )
            wandb.log({"crop_variance_analysis": crop_variance_table})

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

    # 1. Log dataset artifact ONCE before training starts (only if explicitly enabled)
    if not args.no_wandb and base_config.get("log_dataset_artifact", False):
        log_dataset_artifact(base_config)

    if args.sweep:
        with open("sweep.yaml", "r") as f:
            sweep_config = yaml.safe_load(f)
        sweep_id = wandb.sweep(sweep_config, project="image2biomass")
        wandb.agent(sweep_id, function=train)
    else:
        train(config=base_config)
