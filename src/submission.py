import glob
import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

MODEL_PATH = "/kaggle/input/vit-tiny-patch16-224-untrained/vit_tiny_patch16_224_splendid-galaxy-22.pt"


# Initialization
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

model = torch.jit.load(
    MODEL_PATH,
    map_location=device,
)


class SubmissionDataset(Dataset):
    def __init__(self, img_dir: str, transform=None) -> None:
        self.img_dir: str = img_dir
        self.transform = transform

    def __len__(self) -> int:
        img_paths = glob.glob(os.path.join(self.img_dir, "*.jpg"))
        return len(img_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        img_paths = sorted(glob.glob(os.path.join(self.img_dir, "*.jpg")))
        img_path: str = img_paths[idx]
        image: Image.Image = Image.open(img_path).convert("RGB")

        if self.transform:
            image_tensor: torch.Tensor = self.transform(image)

        image_id: str = os.path.basename(img_path).replace(".jpg", "")

        return image_tensor, image_id


dataset = SubmissionDataset(
    img_dir="/kaggle/input/csiro-biomass/test",
    transform=transforms.Compose(  # TODO: verify that these transforms match training
        [
            transforms.Lambda(lambda img: img.crop((500, 0, 1500, 1000))),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
            ),  # ViT often uses different normalization
        ]
    ),
)

dataloader = DataLoader(dataset, batch_size=16, shuffle=True)


# --- MAIN INFERENCE LOOP ---
model.eval()

inference_bar = tqdm(dataloader, desc="Inference")
all_preds = []
SUFFIXES = [
    "__Dry_Clover_g",
    "__Dry_Dead_g",
    "__Dry_Green_g",
    "__Dry_Total_g",
    "__GDM_g",
]

with torch.no_grad():
    for images, images_id in inference_bar:
        images = images.to(device)

        preds = model(images)
        all_preds.append((images_id, preds.cpu()))

# Combine all predictions
with open("./submission.csv", "w") as f:
    for i, (image_ids, preds) in enumerate(all_preds):
        for j, image_id in enumerate(image_ids):
            for k, suffix in enumerate(SUFFIXES):
                if i == 0 and j == 0 and k == 0:
                    f.write("sample_id,target\n")  # Write header
                f.write(f"{image_id}{suffix},{preds[j][k].item()}\n")
