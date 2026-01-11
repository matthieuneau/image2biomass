import os

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import GroupKFold, KFold, LeaveOneGroupOut

import wandb
from models import UnifiedModel

_standard_repr = torch.Tensor.__repr__

# Model configuration mapping
MODEL_CONFIGS = {
    # ConvNeXt models
    "convnext_tiny.fb_in22k": {"last_layer_dim": 768, "image_size": 224},
    "convnext_nano.d1h_in1k": {"last_layer_dim": 640, "image_size": 224},
    "convnext_pico.d1_in1k": {"last_layer_dim": 512, "image_size": 224},
    # Swin Transformer models
    "swin_tiny_patch4_window7_224": {"last_layer_dim": 768, "image_size": 224},
    "swin_s3_tiny_224": {"last_layer_dim": 768, "image_size": 224},
    # DINOv2 model
    "vit_small_patch14_dinov2.lvd142m": {"last_layer_dim": 384, "image_size": 518},
    # Vanilla ViT models
    "vit_tiny_patch16_224": {"last_layer_dim": 192, "image_size": 224},
    "vit_small_patch16_224": {"last_layer_dim": 384, "image_size": 224},
    "vit_small_patch32_224": {"last_layer_dim": 384, "image_size": 224},
    # ResNet models
    "resnet18": {"last_layer_dim": 512, "image_size": 224},
    "resnet34": {"last_layer_dim": 512, "image_size": 224},
    "resnet50": {"last_layer_dim": 2048, "image_size": 224},
    # EfficientNet models
    "efficientnet_b0": {"last_layer_dim": 1280, "image_size": 224},
    "efficientnet_b1": {"last_layer_dim": 1280, "image_size": 240},
    "efficientnet_b2": {"last_layer_dim": 1408, "image_size": 260},
    # MobileNet model
    "mobilenetv3_large_100": {"last_layer_dim": 1280, "image_size": 224},
    # RegNet models
    "regnetx_016": {"last_layer_dim": 912, "image_size": 224},
    "regnetx_032": {"last_layer_dim": 1008, "image_size": 224},
}


def enhanced_repr(self) -> str:
    shape = list(self.shape)
    device = self.device
    dtype = str(self.dtype).replace("torch.", "")  # e.g., 'float32'
    grad = "🔥" if self.requires_grad else "❄️"
    mem = f"{self.element_size() * self.nelement() / 1024:.1f}KB"
    return f"[{shape} | {device} | {dtype} | {grad} | {mem}] -> {_standard_repr(self)}"


# Loss weights for biomass targets
LOSS_WEIGHTS_LIST = [0.1, 0.1, 0.1, 0.2, 0.5]


def compute_per_target_r2(
    y_true: torch.Tensor, y_pred: torch.Tensor, target_cols: list[str]
) -> dict:
    """Compute R² score for each of the 5 targets."""
    r2_dict = {}
    for i, name in enumerate(target_cols):
        y_t = y_true[:, i]
        y_p = y_pred[:, i]
        ss_res = ((y_t - y_p) ** 2).sum()
        ss_tot = ((y_t - y_t.mean()) ** 2).sum()
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else torch.tensor(0.0)
        r2_dict[f"val_R2_{name}"] = r2.item() if isinstance(r2, torch.Tensor) else r2
    return r2_dict


def compute_per_target_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    criterion: nn.Module,
    target_cols: list[str],
) -> dict:
    """Compute MSE loss for each target individually."""
    loss_dict = {}
    for i, name in enumerate(target_cols):
        loss = criterion(y_pred[:, i], y_true[:, i]).mean()
        loss_dict[f"val_loss_{name}"] = loss.item()
    return loss_dict


def compute_residual_stats(
    y_true: torch.Tensor, y_pred: torch.Tensor, target_cols: list[str]
) -> dict:
    """Compute mean and std of residuals for each target."""
    residuals = y_true - y_pred
    stats = {}
    for i, name in enumerate(target_cols):
        res = residuals[:, i]
        stats[f"residual_mean_{name}"] = res.mean().item()
        stats[f"residual_std_{name}"] = res.std().item()
    return stats


def get_gradient_norm(model: nn.Module) -> float:
    """Compute total gradient L2 norm across all parameters."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm**0.5


def create_scatter_plot(
    y_true_np: np.ndarray, y_pred_np: np.ndarray, target_name: str
) -> plt.Figure:
    """Create actual vs predicted scatter plot for a single target."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true_np, y_pred_np, alpha=0.6, edgecolors="black", linewidth=0.5)

    # Add perfect prediction line
    min_val = min(y_true_np.min(), y_pred_np.min())
    max_val = max(y_true_np.max(), y_pred_np.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect")

    ax.set_xlabel(f"Actual {target_name}")
    ax.set_ylabel(f"Predicted {target_name}")
    ax.set_title(f"Actual vs Predicted: {target_name}")
    ax.legend()
    plt.tight_layout()
    return fig


def create_per_target_loss_plot(
    per_target_loss: dict, target_cols: list[str]
) -> plt.Figure:
    """Create a bar plot showing validation loss for each target on the same plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    losses = [per_target_loss[f"val_loss_{name}"] for name in target_cols]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(target_cols)))

    bars = ax.bar(target_cols, losses, color=colors, edgecolor="black", linewidth=0.5)

    # Add value labels on bars
    for bar, loss in zip(bars, losses):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{loss:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xlabel("Target")
    ax.set_ylabel("Validation Loss (MSE)")
    ax.set_title("Per-Target Validation Loss")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    return fig


def create_per_target_r2_plot(
    per_target_r2: dict, target_cols: list[str]
) -> plt.Figure:
    """Create a bar plot showing R² for each target on the same plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    r2_values = [per_target_r2[f"val_R2_{name}"] for name in target_cols]
    colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(target_cols)))

    bars = ax.bar(
        target_cols, r2_values, color=colors, edgecolor="black", linewidth=0.5
    )

    # Add value labels on bars
    for bar, r2 in zip(bars, r2_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{r2:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xlabel("Target")
    ax.set_ylabel("R² Score")
    ax.set_title("Per-Target R² Score")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    return fig


def create_residual_stats_plot(
    residual_stats: dict, target_cols: list[str]
) -> plt.Figure:
    """Create a plot showing mean and std of residuals for each target."""
    fig, ax = plt.subplots(figsize=(10, 6))

    means = [residual_stats[f"residual_mean_{name}"] for name in target_cols]
    stds = [residual_stats[f"residual_std_{name}"] for name in target_cols]

    x = np.arange(len(target_cols))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2, means, width, label="Mean", color="steelblue", edgecolor="black"
    )
    bars2 = ax.bar(
        x + width / 2, stds, width, label="Std", color="coral", edgecolor="black"
    )

    # Add value labels
    for bar, val in zip(bars1, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    for bar, val in zip(bars2, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xlabel("Target")
    ax.set_ylabel("Residual Value")
    ax.set_title("Residual Mean and Std per Target")
    ax.set_xticks(x)
    ax.set_xticklabels(target_cols, rotation=45, ha="right")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    return fig


def log_hard_examples(
    full_dataset,
    val_indices: list,
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    target_cols: list[str],
    n_examples: int = 10,
) -> "wandb.Table":
    """Log N worst predictions as a W&B Table with images."""
    # Compute total weighted error per sample
    weights = torch.tensor(LOSS_WEIGHTS_LIST)
    errors = (weights * (y_true - y_pred) ** 2).sum(dim=1)

    # Get indices of worst predictions
    n_to_log = min(n_examples, len(errors))
    _, worst_indices = torch.topk(errors, n_to_log)

    # Create table columns
    columns = (
        ["image", "total_error"]
        + [f"actual_{t}" for t in target_cols]
        + [f"pred_{t}" for t in target_cols]
        + [f"error_{t}" for t in target_cols]
    )

    table = wandb.Table(columns=columns)

    for idx in worst_indices:
        original_idx = val_indices[idx.item()]

        # Get image path from dataset
        img_name = full_dataset.df.iloc[original_idx]["image_id"] + ".jpg"
        img_path = os.path.join(full_dataset.img_dir, img_name)

        # Build row data
        row = [wandb.Image(img_path), errors[idx].item()]

        # Add actual values
        for i in range(5):
            row.append(y_true[idx, i].item())

        # Add predicted values
        for i in range(5):
            row.append(y_pred[idx, i].item())

        # Add per-target errors
        for i in range(5):
            row.append(((y_true[idx, i] - y_pred[idx, i]) ** 2).item())

        table.add_data(*row)

    return table


def log_crop_variance_analysis(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    image_size: int,
    crop_width: int,
    target_cols: list[str],
    n_images: int = 10,
    n_crops: int = 30,
) -> "wandb.Table":
    """
    Analyze prediction variance due to random crop position.

    For each selected image, runs multiple forward passes with different random crops
    and computes the std of predictions. Only logs Dry_Green_g, Dry_Dead_g, Dry_Total_g.

    Args:
        model: Trained model
        dataset: Dataset with images (should have no transform or raw transform)
        device: Device to run inference on
        image_size: Model input size
        crop_width: Width of random crop
        target_cols: List of target column names
        n_images: Number of images to analyze
        n_crops: Number of random crops per image

    Returns:
        wandb.Table with variance analysis results
    """
    from models import RandomCrop
    from torchvision import transforms
    from PIL import Image

    model.eval()

    # Only analyze these targets
    selected_targets = ["Dry_Green_g", "Dry_Dead_g", "Dry_Total_g"]
    selected_indices = [target_cols.index(t) for t in selected_targets]

    # Determine normalization based on model
    model_name = getattr(model, 'model_name', '')
    if 'dinov2' in model_name.lower():
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    elif 'vit' in model_name.lower() and 'dinov2' not in model_name.lower():
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    else:
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    # Transform with random crop (no fixed seed)
    random_transform = transforms.Compose([
        RandomCrop(crop_width),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # Select random images
    np.random.seed(None)  # Use true randomness
    sample_indices = np.random.choice(len(dataset), size=min(n_images, len(dataset)), replace=False)

    # Build table columns: image, then for each selected target: true, std, std/true ratio
    columns = ["image", "image_id"]
    for t in selected_targets:
        columns.extend([f"true_{t}", f"pred_std_{t}", f"std/true_{t}"])

    table = wandb.Table(columns=columns)

    with torch.no_grad():
        for idx in sample_indices:
            # Get original image path and true values
            img_name = dataset.df.iloc[idx]["image_id"]
            img_path = os.path.join(dataset.img_dir, img_name + ".jpg")
            true_values = dataset.df.iloc[idx][target_cols].values.astype(float)

            # Load raw image
            raw_img = Image.open(img_path).convert("RGB")

            # Run multiple forward passes with different random crops
            all_preds = []
            for _ in range(n_crops):
                img_tensor = random_transform(raw_img).unsqueeze(0).to(device)
                pred = model(img_tensor).cpu().numpy()[0]
                all_preds.append(pred)

            all_preds = np.array(all_preds)  # Shape: (n_crops, 5)
            pred_std = all_preds.std(axis=0)

            # Build row
            row = [wandb.Image(img_path), img_name]

            # Add stats for selected targets only
            for i in selected_indices:
                true_val = float(true_values[i])
                std_val = float(pred_std[i])
                # Avoid division by zero
                ratio = std_val / true_val if true_val > 0 else 0.0
                row.extend([true_val, std_val, ratio])

            table.add_data(*row)

    return table


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
        print("Dataset artifact logged successfully.")


def get_model(config: dict) -> torch.nn.Module:
    """Get model with automatic configuration based on model_name."""
    model_name = config["model_name"]

    # If model has predefined config, use it; otherwise use provided values
    if model_name in MODEL_CONFIGS:
        model_config = MODEL_CONFIGS[model_name]
        # Override config with model-specific values
        config = {**config, **model_config}

    # For backward compatibility, check if last_layer_dim and image_size are provided
    if "last_layer_dim" not in config:
        raise ValueError(f"last_layer_dim not found for model {model_name}")
    if "image_size" not in config:
        raise ValueError(f"image_size not found for model {model_name}")

    return eval(config.get("model_class", "UnifiedModel"))(
        model_name=config["model_name"], last_layer_dim=config["last_layer_dim"]
    )


def upload_model_to_kaggle():
    LOCAL_MODEL_DIR = "./models"

    MODEL_SLUG = "vit-tiny-patch16-224-splendid-galaxy-22"

    # Learn more about naming model variations at
    # https://www.kaggle.com/docs/models#name-model.
    VARIATION_SLUG = "default"

    kagglehub.model_upload(
        handle=f"matthieuneau/{MODEL_SLUG}/pyTorch/{VARIATION_SLUG}",
        local_model_dir=LOCAL_MODEL_DIR,
        version_notes="Update 2026-01-01",
    )


def get_cv_splits(n_samples: int, config: dict, groups: list | None = None):
    """
    Generate cross-validation train/val splits based on config.

    Args:
        n_samples: Total number of samples
        config: Configuration dict with cv_enabled, cv_folds, cv_type
        groups: List of group labels (e.g., states) for region-wise CV

    Yields:
        (fold_idx, train_indices, val_indices, fold_name)
    """
    cv_type = config.get("cv_type", "kfold")
    cv_folds = config.get("cv_folds", 5)

    indices = np.arange(n_samples)

    if cv_type == "region":
        if groups is None:
            raise ValueError("Region-wise CV requires group labels (states)")
        # Use LeaveOneGroupOut for region-wise CV (each region = 1 fold)
        logo = LeaveOneGroupOut()
        unique_groups = list(dict.fromkeys(groups))  # Preserve order
        for fold_idx, (train_idx, val_idx) in enumerate(
            logo.split(indices, groups=groups)
        ):
            # Find which group is left out
            val_group = groups[val_idx[0]]
            fold_name = f"fold_{fold_idx + 1}_val_{val_group}"
            yield fold_idx, train_idx.tolist(), val_idx.tolist(), fold_name
    else:
        # Standard KFold
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(indices)):
            fold_name = f"fold_{fold_idx + 1}_of_{cv_folds}"
            yield fold_idx, train_idx.tolist(), val_idx.tolist(), fold_name
