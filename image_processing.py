import glob
import os

import torch
from PIL import Image
from torchvision import transforms


def preprocess_biomass_images(
    src_dir="./data/train", dest_dir="./data/preprocessed", num_samples=10
):
    # 1. Setup Directories
    os.makedirs(dest_dir, exist_ok=True)

    # 2. Define Transformations (Pretrained ResNet normalization is standard)
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # 3. Get first N images
    img_paths = sorted(glob.glob(os.path.join(src_dir, "*.jpg")))[:num_samples]

    for i, path in enumerate(img_paths):
        with Image.open(path) as img:
            # Get dimensions (assuming 2000x1000)
            w, h = img.size

            # Split image into Left and Right halves (1000x1000 each)
            left_half = img.crop((0, 0, w // 2, h))
            right_half = img.crop((w // 2, 0, w, h))

            # Apply transforms to both
            left_tensor = transform(left_half)  # Shape: [3, 224, 224]
            right_tensor = transform(right_half)  # Shape: [3, 224, 224]

            # Stack into a 2x3x224x224 tensor
            stacked_tensor = torch.stack([left_tensor, right_tensor], dim=0)

            # Save as a .pt file
            filename = os.path.basename(path).replace(".jpg", ".pt")
            save_path = os.path.join(dest_dir, filename)
            torch.save(stacked_tensor, save_path)

            print(
                f"Processed {i + 1}/{num_samples}: {filename} -> Shape: {stacked_tensor.shape}"
            )


import matplotlib.pyplot as plt
import numpy as np


def visualize_stacked_tensor(file_path):
    # 1. Load the tensor (shape: [2, 3, 224, 224])
    # map_location='cpu' ensures it works even if you don't have a GPU active
    tensor = torch.load(file_path, map_location="cpu")

    # 2. Define Un-normalization parameters (Standard ImageNet)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    titles = ["Left Tile (Channel 0)", "Right Tile (Channel 1)"]

    for i in range(2):
        # Extract the i-th tile: [3, 224, 224]
        tile = tensor[i].numpy().transpose(1, 2, 0)  # Convert to [224, 224, 3]

        # Reverse the normalization: (Pixel * std) + mean
        tile = std * tile + mean

        # Clip values to [0, 1] range to avoid Matplotlib warnings
        tile = np.clip(tile, 0, 1)

        axes[i].imshow(tile)
        axes[i].set_title(f"{titles[i]}\nShape: {tensor[i].shape}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    preprocess_biomass_images()
    # To check preprocessing
    # visualize_stacked_tensor("./data/preprocessed/ID1011485656.pt")
