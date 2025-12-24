import matplotlib.pyplot as plt
import numpy as np
import torch


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


# Run for your specific file
visualize_stacked_tensor("./data/preprocessed/ID1011485656.pt")
