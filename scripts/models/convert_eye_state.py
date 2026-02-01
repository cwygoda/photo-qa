#!/usr/bin/env python3
"""
Train and export eye state classifier to safetensors format.

Trains a simple CNN on the CEW (Closed Eyes in the Wild) dataset
to classify whether eyes are open or closed.

Dataset: http://parnec.nuaa.edu.cn/xtan/data/ClosedEyeDatabases.html
"""

import argparse
import hashlib
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from safetensors.torch import save_file
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Input dimensions matching Rust code
EYE_WIDTH = 34
EYE_HEIGHT = 26


class EyeStateClassifier(nn.Module):
    """
    Eye state classifier matching the Rust Candle implementation.

    Architecture:
        Conv1: 1 -> 32 channels, 3x3, padding=1
        MaxPool 2x2: 34x26 -> 17x13
        Conv2: 32 -> 64 channels, 3x3, padding=1
        MaxPool 2x2: 17x13 -> 8x6
        Conv3: 64 -> 128 channels, 3x3, padding=1
        MaxPool 2x2: 8x6 -> 4x3
        FC1: 1536 -> 256
        FC2: 256 -> 1 (logit)
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 4 * 3, 256)  # 1536
        self.fc2 = nn.Linear(256, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Conv1 + ReLU + MaxPool: 34x26 -> 17x13
        x = self.pool(self.relu(self.conv1(x)))
        # Conv2 + ReLU + MaxPool: 17x13 -> 8x6
        x = self.pool(self.relu(self.conv2(x)))
        # Conv3 + ReLU + MaxPool: 8x6 -> 4x3
        x = self.pool(self.relu(self.conv3(x)))
        # Flatten
        x = x.view(x.size(0), -1)
        # FC1 + ReLU
        x = self.relu(self.fc1(x))
        # FC2 (logit)
        x = self.fc2(x)
        return x


class EyeDataset(Dataset):
    """Dataset for eye state classification."""

    def __init__(self, samples: list[tuple[Path, int]], augment: bool = False):
        """
        Args:
            samples: List of (image_path, label) tuples. Label: 0=closed, 1=open
            augment: Apply data augmentation during training
        """
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        # Load and preprocess
        img = Image.open(path).convert("L")
        img = img.resize((EYE_WIDTH, EYE_HEIGHT), Image.Resampling.LANCZOS)

        # Convert to numpy
        arr = np.array(img, dtype=np.float32) / 255.0

        # Augmentation
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                arr = np.fliplr(arr).copy()
            # Random brightness
            arr = arr * (0.8 + random.random() * 0.4)
            arr = np.clip(arr, 0, 1)

        # Convert to tensor [1, H, W]
        tensor = torch.from_numpy(arr).unsqueeze(0)

        return tensor, torch.tensor(label, dtype=torch.float32)


def load_cew_dataset(data_dir: Path) -> list[tuple[Path, int]]:
    """
    Load CEW (Closed Eyes in the Wild) dataset.

    Expected structure:
        data_dir/
            closed_eyes/
                *.jpg
            open_eyes/
                *.jpg

    Or alternatively (original CEW structure):
        data_dir/
            Subjects/
                s0001_00_c.png  (closed)
                s0001_00_o.png  (open)
    """
    samples = []

    # Check for closed_eyes/open_eyes structure
    closed_dir = data_dir / "closed_eyes"
    open_dir = data_dir / "open_eyes"

    if closed_dir.exists() and open_dir.exists():
        for img_path in closed_dir.glob("*"):
            if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                samples.append((img_path, 0))  # closed = 0

        for img_path in open_dir.glob("*"):
            if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                samples.append((img_path, 1))  # open = 1

    # Check for original CEW structure (Subjects folder)
    subjects_dir = data_dir / "Subjects"
    if subjects_dir.exists():
        for img_path in subjects_dir.glob("*"):
            if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                # Naming convention: s0001_00_c.png (closed) or s0001_00_o.png (open)
                name = img_path.stem.lower()
                if name.endswith("_c"):
                    samples.append((img_path, 0))
                elif name.endswith("_o"):
                    samples.append((img_path, 1))

    # Fallback: scan all subdirectories
    if not samples:
        for img_path in data_dir.rglob("*"):
            if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                name = img_path.stem.lower()
                parent = img_path.parent.name.lower()

                if "closed" in name or "closed" in parent or name.endswith("_c"):
                    samples.append((img_path, 0))
                elif "open" in name or "open" in parent or name.endswith("_o"):
                    samples.append((img_path, 1))

    return samples


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    device: str = "cpu",
) -> nn.Module:
    """Train the eye state classifier."""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.BCEWithLogitsLoss()

    best_val_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            inputs = inputs.to(device)
            labels = labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device).unsqueeze(1)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    return model


def convert_to_safetensors(model: nn.Module) -> dict:
    """Convert model weights to safetensors format matching Rust naming."""
    weights = {}

    state_dict = model.state_dict()
    for key, value in state_dict.items():
        weights[key] = value.float()

    return weights


def compute_sha256(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Train and export eye state classifier")
    parser.add_argument(
        "--data-dir", "-d",
        type=Path,
        help="Path to CEW dataset directory"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("eye_state.safetensors"),
        help="Output safetensors file path"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size"
    )
    parser.add_argument(
        "--random-init",
        action="store_true",
        help="Save random initialization (no training)"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Training device (cpu, cuda, mps)"
    )
    args = parser.parse_args()

    model = EyeStateClassifier()

    if args.random_init:
        print("Saving randomly initialized eye state classifier...")
    else:
        if not args.data_dir:
            print("Error: --data-dir required for training")
            print("Download CEW dataset from: http://parnec.nuaa.edu.cn/xtan/data/ClosedEyeDatabases.html")
            return

        print(f"Loading dataset from {args.data_dir}...")
        samples = load_cew_dataset(args.data_dir)

        if not samples:
            print("Error: No samples found in dataset directory")
            print("Expected structure:")
            print("  data_dir/closed_eyes/*.jpg")
            print("  data_dir/open_eyes/*.jpg")
            print("Or:")
            print("  data_dir/Subjects/*_c.png (closed)")
            print("  data_dir/Subjects/*_o.png (open)")
            return

        closed = sum(1 for _, label in samples if label == 0)
        opened = sum(1 for _, label in samples if label == 1)
        print(f"Found {len(samples)} samples: {closed} closed, {opened} open")

        # Split dataset
        train_samples, val_samples = train_test_split(
            samples, test_size=0.2, random_state=42, stratify=[s[1] for s in samples]
        )

        train_dataset = EyeDataset(train_samples, augment=True)
        val_dataset = EyeDataset(val_samples, augment=False)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

        print(f"\nTraining on {len(train_samples)} samples, validating on {len(val_samples)}")
        model = train_model(model, train_loader, val_loader, epochs=args.epochs, device=args.device)

    # Convert and save
    weights = convert_to_safetensors(model)

    print(f"\nSaving to {args.output}...")
    save_file(weights, args.output)

    sha256 = compute_sha256(args.output)
    file_size = args.output.stat().st_size

    print(f"\nConversion complete!")
    print(f"  File: {args.output}")
    print(f"  Size: {file_size:,} bytes ({file_size / 1024:.2f} KB)")
    print(f"  SHA256: {sha256}")
    print(f"\nWeight keys:")
    for key in sorted(weights.keys()):
        print(f"  {key}: {list(weights[key].shape)}")


if __name__ == "__main__":
    main()
