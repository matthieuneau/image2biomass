import timm
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import v2


class CenterCrop:
    """Picklable center crop transform for multiprocessing DataLoader."""
    def __init__(self, crop_width: int = 1000):
        self.crop_width = crop_width

    def __call__(self, img):
        left = (img.width - self.crop_width) // 2
        right = left + self.crop_width
        return img.crop((left, 0, right, img.height))


class RandomCrop:
    """Picklable random crop transform that takes a random horizontal slice.

    Similar to CenterCrop but the crop position is randomized.
    Default crop width is 1000px (same as CenterCrop: 1500-500).
    """
    def __init__(self, crop_width: int = 1000):
        self.crop_width = crop_width

    def __call__(self, img):
        max_left = img.width - self.crop_width
        if max_left <= 0:
            return img  # Image is smaller than crop width, return as-is
        left = torch.randint(0, max_left + 1, (1,)).item()
        right = left + self.crop_width
        return img.crop((left, 0, right, img.height))


class TileCrop:
    """Extract a square tile at a specific (x, y) position (for TTA with 2D grid)."""
    def __init__(self, tile_size: int, x: int, y: int):
        """
        Args:
            tile_size: Size of the square tile to extract
            x: Left position of the tile (0-indexed)
            y: Top position of the tile (0-indexed)
        """
        self.tile_size = tile_size
        self.x = x
        self.y = y

    def __call__(self, img):
        left = self.x
        top = self.y
        right = min(left + self.tile_size, img.width)
        bottom = min(top + self.tile_size, img.height)
        return img.crop((left, top, right, bottom))


def _get_activation(name: str) -> nn.Module:
    """Returns activation module by name."""
    activations = {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(),
        "leaky_relu": nn.LeakyReLU(0.1),
        "softplus": nn.Softplus(),
    }
    return activations.get(name.lower(), nn.ReLU())


class Backbone(nn.Module):
    """Feature extractor wrapping a timm model."""

    def __init__(self, model_name: str, pretrained: bool = True):
        super().__init__()
        self.model_name = model_name
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Regressor(nn.Module):
    """Configurable regression head for biomass prediction."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.2,
        activation: str = "relu",
        output_activation: str = "relu",
        normalization: str = "none",
        num_outputs: int = 5,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.layers = self._build_layers(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            output_activation=output_activation,
            normalization=normalization,
            num_outputs=num_outputs,
        )

    def _build_layers(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        activation: str,
        output_activation: str,
        normalization: str,
        num_outputs: int,
    ) -> nn.Sequential:
        """Builds the regression layers."""
        layers = []
        current_dim = in_dim

        # Hidden layers
        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))

            # Normalization (before activation)
            if normalization == "batch":
                layers.append(nn.BatchNorm1d(hidden_dim))
            elif normalization == "layer":
                layers.append(nn.LayerNorm(hidden_dim))

            # Activation
            layers.append(_get_activation(activation))

            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            current_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(current_dim, num_outputs))

        # Output activation (ensure non-negative)
        layers.append(_get_activation(output_activation))

        return nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


class UnifiedModel(nn.Module):
    """Unified model class that composes a Backbone and Regressor."""

    def __init__(
        self,
        model_name: str,
        last_layer_dim: int,
        pretrained: bool = True,
        head_hidden_dim: int = 128,
        head_num_layers: int = 1,
        head_dropout: float = 0.2,
        head_activation: str = "relu",
        head_output_activation: str = "relu",
        head_normalization: str = "none",
    ):
        super().__init__()
        self.model_name = model_name

        # 1. Feature Extractor
        self.backbone = Backbone(model_name, pretrained=pretrained)

        # 2. Regression Head
        self.regressor = Regressor(
            in_dim=last_layer_dim,
            hidden_dim=head_hidden_dim,
            num_layers=head_num_layers,
            dropout=head_dropout,
            activation=head_activation,
            output_activation=head_output_activation,
            normalization=head_normalization,
        )

    @staticmethod
    def get_transforms_static(model_name: str, image_size: int, crop_type: str = "center", crop_width: int = 1000):
        """Returns the appropriate transforms based on model architecture.

        Args:
            model_name: Name of the model (for normalization selection)
            image_size: Target image size after resize
            crop_type: "center" for CenterCrop, "random" for RandomCrop
            crop_width: Width of the crop in pixels (default 1000)
        """
        # Determine normalization based on model type
        if 'dinov2' in model_name.lower():
            # DINOv2 models use ImageNet normalization
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        elif 'vit' in model_name.lower() and 'dinov2' not in model_name.lower():
            # Standard ViT models often use different normalization
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
        else:
            # ConvNets (ResNet, EfficientNet, ConvNeXt, etc.) use ImageNet stats
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

        # Select crop transform based on crop_type
        crop_transform = RandomCrop(crop_width) if crop_type == "random" else CenterCrop(crop_width)

        return transforms.Compose([
            crop_transform,
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def get_transforms(self, image_size: int, crop_type: str = "center", crop_width: int = 1000):
        """Instance method that calls the static method"""
        return UnifiedModel.get_transforms_static(self.model_name, image_size, crop_type, crop_width)

    @staticmethod
    def get_train_transforms_static(model_name: str, image_size: int, crop_type: str = "center", crop_width: int = 1000):
        """Returns training transforms with data augmentation (2025 standard approach).

        Uses torchvision.transforms.v2 with:
        - RandomResizedCrop: Varies scale and aspect ratio
        - RandomHorizontalFlip: Natural for top-down pasture views
        - ColorJitter: Handles varied outdoor lighting conditions
        - TrivialAugmentWide: Automatic augmentation policy (simpler than RandAugment)

        Args:
            model_name: Name of the model (for normalization selection)
            image_size: Target image size after resize
            crop_type: "center" for CenterCrop, "random" for RandomCrop
            crop_width: Width of the crop in pixels (default 1000)
        """
        # Determine normalization based on model type
        if 'dinov2' in model_name.lower():
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        elif 'vit' in model_name.lower() and 'dinov2' not in model_name.lower():
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
        else:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

        # Select crop transform based on crop_type
        crop_transform = RandomCrop(crop_width) if crop_type == "random" else CenterCrop(crop_width)

        return v2.Compose([
            crop_transform,
            v2.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(0.8, 1.0),      # Crop 80-100% of the image
                ratio=(0.9, 1.1),      # Near-square aspect ratio
                antialias=True,
            ),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
            ),
            v2.TrivialAugmentWide(),  # Automatic augmentation policy
            v2.ToImage(),
            v2.ToDtype(dtype=torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        ])

    def get_train_transforms(self, image_size: int, crop_type: str = "center", crop_width: int = 1000):
        """Instance method for training transforms with augmentation"""
        return UnifiedModel.get_train_transforms_static(self.model_name, image_size, crop_type, crop_width)

    def forward(self, x):
        features = self.backbone(x)
        biomass = self.regressor(features)
        return biomass

    def predict_with_tta(
        self,
        raw_image,
        image_size: int,
        tile_size: int,
        tta_tiles: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Perform inference with Test Time Augmentation using a 2D grid of tiles.

        Images are 2000x1000, so we use a grid with 2x more columns than rows.
        For tta_tiles=18, this creates a 3x6 grid (3 rows, 6 columns).

        Args:
            raw_image: PIL Image (raw, untransformed, expected 2000x1000)
            image_size: Model input size (e.g., 224)
            tile_size: Size of each square tile to extract
            tta_tiles: Total number of tiles (should be rows * cols where cols = 2 * rows)
            device: Device to run inference on

        Returns:
            Averaged predictions across all tiles (shape: [5])
        """
        import math

        # Determine normalization based on model type
        if 'dinov2' in self.model_name.lower():
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        elif 'vit' in self.model_name.lower() and 'dinov2' not in self.model_name.lower():
            mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        else:
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

        img_width, img_height = raw_image.width, raw_image.height

        # Calculate grid dimensions (2x more columns than rows for 2000x1000 images)
        # tta_tiles = rows * cols, cols = 2 * rows => tta_tiles = 2 * rows^2
        # rows = sqrt(tta_tiles / 2)
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

        # Create transforms and run inference for each tile
        all_preds = []
        was_training = self.training
        self.eval()

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
                    pred = self.forward(img_tensor)
                    all_preds.append(pred)

        # Restore training mode if it was set
        if was_training:
            self.train()

        # Average predictions across all tiles
        all_preds = torch.cat(all_preds, dim=0)  # Shape: (n_rows * n_cols, 5)
        return all_preds.mean(dim=0)  # Shape: (5,)


class EnsembleModel(nn.Module):
    """Ensemble model that combines multiple backbones with a shared regression head.

    Loads pre-trained backbones (saved as TorchScript), freezes them, concatenates
    their features, and trains a new regression head on the combined features.
    """

    def __init__(
        self,
        backbone_paths: list[str],
        feature_dims: list[int],
        head_hidden_dim: int = 256,
        head_num_layers: int = 2,
        head_dropout: float = 0.3,
        head_activation: str = "relu",
        head_output_activation: str = "relu",
        head_normalization: str = "layer",
        freeze_backbones: bool = True,
    ):
        """
        Args:
            backbone_paths: List of paths to saved backbone .pt files
            feature_dims: List of feature dimensions for each backbone (must match order)
            head_hidden_dim: Hidden dimension for the ensemble regressor
            head_num_layers: Number of hidden layers in ensemble regressor
            head_dropout: Dropout rate for ensemble regressor
            head_activation: Activation function for hidden layers
            head_output_activation: Activation function for output (ensures non-negative)
            head_normalization: Normalization type ("none", "batch", "layer")
            freeze_backbones: Whether to freeze backbone weights (recommended)
        """
        super().__init__()

        if len(backbone_paths) != len(feature_dims):
            raise ValueError("backbone_paths and feature_dims must have same length")

        # Load backbones
        self.backbones = nn.ModuleList()
        for path in backbone_paths:
            backbone = torch.jit.load(path)
            self.backbones.append(backbone)

        # Freeze backbones if requested
        if freeze_backbones:
            for backbone in self.backbones:
                for param in backbone.parameters():
                    param.requires_grad = False

        # Total feature dimension is sum of all backbone outputs
        total_feature_dim = sum(feature_dims)

        # Create ensemble regression head
        self.regressor = Regressor(
            in_dim=total_feature_dim,
            hidden_dim=head_hidden_dim,
            num_layers=head_num_layers,
            dropout=head_dropout,
            activation=head_activation,
            output_activation=head_output_activation,
            normalization=head_normalization,
        )

        # Store metadata
        self.feature_dims = feature_dims
        self.freeze_backbones = freeze_backbones

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: extract features from all backbones, concatenate, predict."""
        features = []
        for backbone in self.backbones:
            feat = backbone(x)
            features.append(feat)

        # Concatenate features from all backbones
        combined = torch.cat(features, dim=1)

        # Pass through ensemble regressor
        return self.regressor(combined)

    @property
    def num_backbones(self) -> int:
        return len(self.backbones)

    @property
    def total_feature_dim(self) -> int:
        return sum(self.feature_dims)

