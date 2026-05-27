# Copyright 2026 HOUMO AI
#
# File: train_dinov3_classifier_head.py
# Description:
#   Script to train a linear classifier head for DINOv3 base model on ImageNet dataset.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision.datasets import ImageFolder
from transformers import AutoImageProcessor, AutoModel

DEFAULT_MODEL_DIR = "./dinov3-vitb16-pretrain-lvd1689m"
DEFAULT_DATASET_DIR = "/YOUR_DATASET_PATH/imagenet/val"
DEFAULT_WORKDIR = Path("YOUR_WORKDIR/DINOv3")
DEFAULT_FEATURE_CACHE_PATH = DEFAULT_WORKDIR / "dinov3_imagenet_val_features.pt"
DEFAULT_TRACE_PATH = DEFAULT_WORKDIR / "dinov3-vitb16-imagenet-linear-head.pt"
DEFAULT_LABELS_PATH = DEFAULT_WORKDIR / "dinov3-vitb16-imagenet-linear-head.labels.json"


class LinearClassifierHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(in_dim, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)


class ImageFolderNoTransform(Dataset):
    def __init__(self, root: str, max_samples: int | None = None, seed: int = 42) -> None:
        self.dataset = ImageFolder(root=root)
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx
        self.samples = self.dataset.samples

        if max_samples is None or max_samples >= len(self.dataset):
            self.indices = list(range(len(self.dataset)))
        else:
            rng = random.Random(seed)
            indices = list(range(len(self.dataset)))
            rng.shuffle(indices)
            self.indices = sorted(indices[:max_samples])

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[Image.Image, int, str]:
        real_index = self.indices[index]
        path, label = self.samples[real_index]
        image = Image.open(path).convert("RGB")
        return image, label, path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="First extract DINOv3 features offline, then train a linear classifier head, and export it as a JIT trace file."
    )
    parser.add_argument("--model-dir", type=str, default=DEFAULT_MODEL_DIR, help="DINOv3 model directory")
    parser.add_argument("--dataset-dir", type=str, default=DEFAULT_DATASET_DIR, help="ImageFolder dataset directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training the linear head")
    parser.add_argument("--extract-batch-size", type=int, default=128, help="Batch size for offline feature extraction")
    parser.add_argument("--train-batch-size", type=int, default=1024, help="Batch size for training the linear head")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of DataLoader workers")
    parser.add_argument("--lr", type=float, default=5e-2, help="Learning rate for the linear head")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--device", type=str, default="auto", help="auto/cpu/cuda/cuda:0")
    parser.add_argument("--feature-cache-path", type=str, default=str(DEFAULT_FEATURE_CACHE_PATH), help="Path to cache offline features")
    parser.add_argument("--force-reextract", action="store_true", help="Force re-extraction of features")
    parser.add_argument("--trace-path", type=str, default=str(DEFAULT_TRACE_PATH), help="Path to save the traced head")
    parser.add_argument("--labels-path", type=str, default=str(DEFAULT_LABELS_PATH), help="Path to save the label mapping JSON")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split ratio from cached features")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit the number of samples for debugging")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def build_collate_fn(processor: AutoImageProcessor):
    def collate_fn(batch: Sequence[tuple[Image.Image, int, str]]) -> tuple[dict[str, torch.Tensor], torch.Tensor, list[str]]:
        images, labels, paths = zip(*batch)
        inputs = processor(images=list(images), return_tensors="pt")
        return inputs, torch.tensor(labels, dtype=torch.long), list(paths)

    return collate_fn


def extract_backbone_features(backbone: AutoModel, pixel_values: torch.Tensor) -> torch.Tensor:
    outputs = backbone(pixel_values=pixel_values)
    features = getattr(outputs, "pooler_output", None)
    if features is None:
        features = outputs.last_hidden_state[:, 0]
    return F.normalize(features, dim=1)


def extract_and_cache_features(
    model_dir: Path,
    dataset_dir: Path,
    cache_path: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    max_samples: int | None,
    seed: int,
    force_reextract: bool,
) -> dict:
    if cache_path.exists() and not force_reextract:
        print(f"Loading cached features from: {cache_path}")
        return torch.load(cache_path, map_location="cpu")

    dataset = ImageFolderNoTransform(str(dataset_dir), max_samples=max_samples, seed=seed)
    processor = AutoImageProcessor.from_pretrained(model_dir)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=build_collate_fn(processor),
    )

    backbone = AutoModel.from_pretrained(model_dir).to(device)
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad = False

    feature_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []
    sample_paths: list[str] = []

    with torch.inference_mode():
        for step, (inputs, labels, paths) in enumerate(data_loader, start=1):
            pixel_values = inputs["pixel_values"].to(device, non_blocking=True)
            features = extract_backbone_features(backbone, pixel_values)
            feature_batches.append(features.cpu())
            label_batches.append(labels.cpu())
            sample_paths.extend(paths)
            if step % 20 == 0:
                print(f"Feature extraction step {step}/{len(data_loader)}")

    feature_tensor = torch.cat(feature_batches, dim=0)
    label_tensor = torch.cat(label_batches, dim=0)
    payload = {
        "features": feature_tensor,
        "labels": label_tensor,
        "classes": dataset.classes,
        "class_to_idx": dataset.class_to_idx,
        "paths": sample_paths,
        "hidden_size": int(feature_tensor.shape[1]),
        "dataset_dir": str(dataset_dir),
        "model_dir": str(model_dir),
        "max_samples": max_samples,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    print(f"Saved cached features to: {cache_path}")
    return payload


def stratified_split(labels: torch.Tensor, val_split: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0.0 < val_split < 1.0:
        raise ValueError("val_split must be between (0, 1).")

    class_to_indices: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels.tolist()):
        class_to_indices[int(label)].append(index)

    rng = random.Random(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    for indices in class_to_indices.values():
        rng.shuffle(indices)
        if len(indices) == 1:
            train_indices.extend(indices)
            continue
        val_count = max(1, int(round(len(indices) * val_split)))
        val_count = min(val_count, len(indices) - 1)
        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])

    if not val_indices:
        raise ValueError("Validation set is empty. Increase the number of samples or the val_split ratio.")
    return torch.tensor(train_indices, dtype=torch.long), torch.tensor(val_indices, dtype=torch.long)


def build_feature_loader(features: torch.Tensor, labels: torch.Tensor, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(features, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def run_epoch(
    head: LinearClassifierHead,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, float]:
    is_train = optimizer is not None
    head.train(is_train)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    context = torch.enable_grad() if is_train else torch.inference_mode()
    with context:
        for features, labels in data_loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = head(features)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += labels.size(0)

    return total_loss / max(total_samples, 1), total_correct / max(total_samples, 1)


def trace_and_save_head(head: LinearClassifierHead, hidden_size: int, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    head_cpu = head.eval().cpu()
    example = torch.randn(1, hidden_size, dtype=torch.float32)
    traced = torch.jit.trace(head_cpu, example)
    traced.save(str(output_path))


def save_labels(classes: list[str], class_to_idx: dict[str, int], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "classes": classes,
        "class_to_idx": class_to_idx,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    model_dir = Path(args.model_dir)
    dataset_dir = Path(args.dataset_dir)
    cache_path = Path(args.feature_cache_path)
    trace_path = Path(args.trace_path)
    labels_path = Path(args.labels_path)

    cache = extract_and_cache_features(
        model_dir=model_dir,
        dataset_dir=dataset_dir,
        cache_path=cache_path,
        device=device,
        batch_size=args.extract_batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        seed=args.seed,
        force_reextract=args.force_reextract,
    )

    features = cache["features"].float()
    labels = cache["labels"].long()
    classes = list(cache["classes"])
    class_to_idx = dict(cache["class_to_idx"])
    hidden_size = int(cache["hidden_size"])

    train_indices, val_indices = stratified_split(labels, val_split=args.val_split, seed=args.seed)
    train_loader = build_feature_loader(features[train_indices], labels[train_indices], args.train_batch_size, shuffle=True)
    val_loader = build_feature_loader(features[val_indices], labels[val_indices], args.train_batch_size, shuffle=False)

    head = LinearClassifierHead(hidden_size, len(classes)).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_state: dict[str, torch.Tensor] | None = None
    best_val_acc = -1.0

    print(f"Device: {device}")
    print(f"Dataset dir: {dataset_dir}")
    print(f"Feature cache: {cache_path}")
    print(f"Samples: {len(features)} | Classes: {len(classes)} | Hidden size: {hidden_size}")
    print("Note: One epoch is usually insufficient for full convergence on a 1000-class task. Observe if train/val loss continues to decrease.")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            head=head,
            data_loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )
        val_loss, val_acc = run_epoch(
            head=head,
            data_loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
        )
        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc * 100:.2f}% | "
            f"val_loss={val_loss:.4f} val_acc={val_acc * 100:.2f}%"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}

    if best_state is not None:
        head.load_state_dict(best_state)

    trace_and_save_head(head, hidden_size, trace_path)
    save_labels(classes, class_to_idx, labels_path)
    print(f"Saved traced head to: {trace_path}")
    print(f"Saved label mapping to: {labels_path}")


if __name__ == "__main__":
    main()
