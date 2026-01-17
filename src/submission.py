"""
Self-contained inference script for Kaggle submission.
Uses regular tiling for Test Time Augmentation (TTA).
"""

import glob
import math
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_PATH = "/kaggle/input/vit-tiny-patch16-224v40-0-68r2val/model_rd1jyrs2.pt"
TEST_DIR = "/kaggle/input/csiro-biomass/test"
OUTPUT_PATH = "./submission.csv"

# Model settings (must match training)
MODEL_NAME = "vit_tiny_patch16_224"  # Used to determine normalization
IMAGE_SIZE = 224  # Model input size

# Test Time Augmentation (TTA) settings
TTA = True        # Enable TTA during inference
TTA_TILES = 18    # Number of tiles in 2D grid (18 = 3x6 grid for 2000x1000 images)
TILE_SIZE = 500   # Size of each square tile

# =============================================================================
# TRANSFORMS (self-contained for Kaggle)
# =============================================================================


class TileCrop:
    """Extract a square tile at a specific (x, y) position."""

    def __init__(self, tile_size: int, x: int, y: int):
        self.tile_size = tile_size
        self.x = x
        self.y = y

    def __call__(self, img):
        left = self.x
        top = self.y
        right = min(left + self.tile_size, img.width)
        bottom = min(top + self.tile_size, img.height)
        return img.crop((left, top, right, bottom))


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_normalization(model_name: str) -> tuple[list[float], list[float]]:
    """Get normalization parameters based on model type."""
    if "dinov2" in model_name.lower():
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    elif "vit" in model_name.lower() and "dinov2" not in model_name.lower():
        return [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    else:
        # ConvNets (ResNet, EfficientNet, ConvNeXt, etc.) use ImageNet stats
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def predict_with_tta(
    model,
    raw_image: Image.Image,
    model_name: str,
    image_size: int,
    tile_size: int,
    tta_tiles: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Perform inference with TTA using a regular 2D grid of tiles.

    Images are 2000x1000, so we use a grid with 2x more columns than rows.
    For tta_tiles=18, this creates a 3x6 grid (3 rows, 6 columns).
    """
    mean, std = get_normalization(model_name)
    img_width, img_height = raw_image.width, raw_image.height

    # Calculate grid dimensions (2x more columns than rows for 2000x1000 images)
    n_rows = max(1, int(math.sqrt(tta_tiles / 2)))
    n_cols = max(1, tta_tiles // n_rows)

    # Calculate evenly spaced tile positions
    max_x = max(0, img_width - tile_size)
    max_y = max(0, img_height - tile_size)

    if n_cols == 1:
        x_positions = [max_x // 2]
    else:
        x_positions = [int(i * max_x / (n_cols - 1)) for i in range(n_cols)]

    if n_rows == 1:
        y_positions = [max_y // 2]
    else:
        y_positions = [int(i * max_y / (n_rows - 1)) for i in range(n_rows)]

    # Run inference for each tile
    all_preds = []
    with torch.no_grad():
        for y in y_positions:
            for x in x_positions:
                tile_transform = transforms.Compose([
                    TileCrop(tile_size, x, y),
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ])
                img_tensor = tile_transform(raw_image).unsqueeze(0).to(device)
                pred = model(img_tensor)
                all_preds.append(pred)

    # Average predictions across all tiles
    all_preds = torch.cat(all_preds, dim=0)
    return all_preds.mean(dim=0)


# =============================================================================
# DATASET
# =============================================================================


class SubmissionDataset(Dataset):
    """Dataset for inference - returns raw images."""

    def __init__(self, img_dir: str) -> None:
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> tuple[Image.Image, str]:
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image_id = os.path.basename(img_path).replace(".jpg", "")
        return image, image_id


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Device setup
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # Load model
    model = torch.jit.load(MODEL_PATH, map_location=device)
    model.eval()

    # Column suffixes for submission (must match expected order)
    SUFFIXES = [
        "__Dry_Green_g",
        "__Dry_Dead_g",
        "__Dry_Clover_g",
        "__GDM_g",
        "__Dry_Total_g",
    ]

    # Dataset
    dataset = SubmissionDataset(img_dir=TEST_DIR)
    all_results = []

    if TTA:
        # TTA mode: use regular grid of tiles
        n_rows = max(1, int(math.sqrt(TTA_TILES / 2)))
        n_cols = max(1, TTA_TILES // n_rows)
        print(f"Running inference with TTA ({n_rows}x{n_cols} grid, {TILE_SIZE}px tiles)")

        for raw_image, image_id in tqdm(dataset, desc="TTA Inference"):
            pred = predict_with_tta(
                model=model,
                raw_image=raw_image,
                model_name=MODEL_NAME,
                image_size=IMAGE_SIZE,
                tile_size=TILE_SIZE,
                tta_tiles=TTA_TILES,
                device=device,
            )
            all_results.append((image_id, pred.cpu()))
    else:
        # Standard mode: single center tile
        print(f"Running inference without TTA (center {TILE_SIZE}px tile)")
        mean, std = get_normalization(MODEL_NAME)
        center_transform = transforms.Compose([
            TileCrop(TILE_SIZE, x=750, y=250),  # Center of 2000x1000 image
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        with torch.no_grad():
            for raw_image, image_id in tqdm(dataset, desc="Inference"):
                img_tensor = center_transform(raw_image).unsqueeze(0).to(device)
                pred = model(img_tensor).squeeze(0)
                all_results.append((image_id, pred.cpu()))

    # Write submission file
    print(f"Writing results to {OUTPUT_PATH}")
    with open(OUTPUT_PATH, "w") as f:
        f.write("sample_id,target\n")
        for image_id, pred in all_results:
            for k, suffix in enumerate(SUFFIXES):
                f.write(f"{image_id}{suffix},{pred[k].item()}\n")

    print(f"Done! Processed {len(all_results)} images.")
