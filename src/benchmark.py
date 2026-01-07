"""
Benchmark script to diagnose training bottlenecks.
Run on both local Mac and remote GPU to compare.
"""

import time
import torch
import yaml
from torch.utils.data import DataLoader
from image_processing import BiomassDataset
from utils import get_model, MODEL_CONFIGS

def benchmark():
    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    # Check CUDA specifics
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")

    # Setup model
    model = get_model(config).to(device)
    model_name = config["model_name"]
    image_size = MODEL_CONFIGS.get(model_name, {}).get("image_size", 224)

    # Setup data
    dataset = BiomassDataset(
        csv_path="./data/y_train.csv",
        img_dir="./data/train",
        transform=model.get_transforms(image_size),
    )

    # Test different num_workers
    print("\n" + "="*60)
    print("1. DATALOADER BENCHMARK (varying num_workers)")
    print("="*60)

    for num_workers in [0, 2, 4, 8]:
        loader = DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(num_workers > 0),
        )

        # Warm up
        iter_loader = iter(loader)
        next(iter_loader)

        # Time 10 batches
        start = time.perf_counter()
        for i, (images, labels) in enumerate(loader):
            if i >= 10:
                break
        data_time = time.perf_counter() - start
        print(f"  num_workers={num_workers}: {data_time:.3f}s for 10 batches ({data_time/10*1000:.1f}ms/batch)")

    # Test data loading + transfer
    print("\n" + "="*60)
    print("2. HOST-TO-DEVICE TRANSFER BENCHMARK")
    print("="*60)

    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)

    load_times = []
    transfer_times = []

    for i, (images, labels) in enumerate(loader):
        if i >= 20:
            break

        # Time transfer to device
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        transfer_times.append(time.perf_counter() - start)

    avg_transfer = sum(transfer_times) / len(transfer_times) * 1000
    print(f"  Avg transfer time: {avg_transfer:.2f}ms/batch")

    # Test forward pass
    print("\n" + "="*60)
    print("3. FORWARD PASS BENCHMARK")
    print("="*60)

    model.eval()
    dummy = torch.randn(config["batch_size"], 3, image_size, image_size, device=device)

    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    forward_times = []
    for _ in range(20):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        forward_times.append(time.perf_counter() - start)

    avg_forward = sum(forward_times) / len(forward_times) * 1000
    print(f"  Avg forward pass: {avg_forward:.2f}ms/batch")

    # Test full training step
    print("\n" + "="*60)
    print("4. FULL TRAINING STEP BENCHMARK")
    print("="*60)

    model.train()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4, pin_memory=(device.type == "cuda"))

    step_times = []
    data_times = []
    compute_times = []

    if device.type == "cuda":
        torch.cuda.synchronize()

    data_start = time.perf_counter()
    for i, (images, labels) in enumerate(loader):
        if i >= 20:
            break

        data_end = time.perf_counter()
        data_times.append(data_end - data_start)

        # Compute
        compute_start = time.perf_counter()
        images, labels = images.to(device), labels.to(device).float()
        preds = model(images)
        loss = criterion(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize()
        compute_end = time.perf_counter()
        compute_times.append(compute_end - compute_start)
        step_times.append(compute_end - data_start)

        data_start = time.perf_counter()

    avg_data = sum(data_times) / len(data_times) * 1000
    avg_compute = sum(compute_times) / len(compute_times) * 1000
    avg_step = sum(step_times) / len(step_times) * 1000

    print(f"  Avg data loading: {avg_data:.2f}ms")
    print(f"  Avg compute (fwd+bwd+opt): {avg_compute:.2f}ms")
    print(f"  Avg total step: {avg_step:.2f}ms")
    print(f"  Data loading overhead: {avg_data/avg_step*100:.1f}%")

    # Summary
    print("\n" + "="*60)
    print("DIAGNOSIS")
    print("="*60)
    if avg_data > avg_compute:
        print("  -> BOTTLENECK: Data loading is slower than compute!")
        print("     Try: increase num_workers, use faster storage, or cache data in RAM")
    else:
        print("  -> Data loading is NOT the bottleneck")
        if avg_transfer > 5:
            print("  -> BOTTLENECK: Host-to-device transfer is slow")
            print("     This could indicate PCIe issues or virtualization overhead")
        else:
            print("  -> Check GPU utilization with nvidia-smi")


if __name__ == "__main__":
    benchmark()
