"""Controlled v2 training for the YOLO11 + MediaPipe Hands LSTM."""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset

from scripts.train_yolo_pose_hands_lstm import (
    content_safe_view,
    sha256,
)
from src.models.yolo_pose_hands_lstm_v2 import (
    FeatureNormalizer,
    LSTMClassifier,
)

META = ROOT / "data" / "metadata"
FEATURE_ROOT = ROOT / "data" / "keypoints_yolo_pose_hands"
OUT = ROOT / "models" / "yolo_pose_hands_lstm_v2"
REPORT = ROOT / "reports" / "yolo11_migration" / "controlled_training_v2_report.md"
SEED = 42
BATCH_SIZE = 32
MAX_EPOCHS = 100
PATIENCE = 15
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 0.0001
DEVELOPMENT_SPLITS = ("train", "validation")


def load_development_rows() -> pd.DataFrame:
    metadata = pd.read_csv(FEATURE_ROOT / "extraction_metadata.csv", dtype={"video_id": str})
    manifest_parts = []
    for split in DEVELOPMENT_SPLITS:
        path = META / f"{split}_signer_disjoint.csv"
        part = pd.read_csv(path, dtype={"video_id": str})
        part["original_split"] = split
        manifest_parts.append(part)
    manifest = pd.concat(manifest_parts, ignore_index=True)
    keys = ["original_split", "dataset", "video_id"]
    metadata = metadata[metadata.original_split.isin(DEVELOPMENT_SPLITS)]
    rows = metadata.merge(manifest, on=keys, suffixes=("_meta", "_manifest"), validate="one_to_one")
    if len(rows) != len(metadata):
        raise ValueError("Development metadata does not fully match train/validation manifests.")
    for field in ("gloss", "signer_id"):
        left, right = f"{field}_meta", f"{field}_manifest"
        if (rows[left].astype(str) != rows[right].astype(str)).any():
            raise ValueError(f"Metadata disagrees with development manifest for {field}.")
    rows["gloss"] = rows["gloss_manifest"]
    rows["signer_id"] = rows["signer_id_manifest"]
    rows["feature_path"] = rows.feature_file.map(lambda p: (ROOT / str(p)).resolve())
    rows["video_path_abs"] = rows.video_path.map(lambda p: (ROOT / str(p)).resolve())
    return rows


def validate_development_view(rows: pd.DataFrame, mapping: dict[str, int]) -> None:
    expected = set(mapping)
    for split in DEVELOPMENT_SPLITS:
        part = rows[rows.original_split == split]
        if expected - set(part.gloss.astype(str)):
            raise ValueError(f"{split} loses class coverage after content exclusions.")
        for row in part.itertuples(index=False):
            with np.load(row.feature_path, allow_pickle=False) as data:
                arrays = (data["raw"], data["presence"], data["lstm"])
                if arrays[0].shape != (30, 177) or arrays[1].shape != (30, 3) or arrays[2].shape != (30, 180):
                    raise ValueError(f"Unexpected feature shapes: {row.feature_file}")
                if not all(np.isfinite(array).all() for array in arrays):
                    raise ValueError(f"Non-finite feature values: {row.feature_file}")
    signers = {
        split: set(map(tuple, rows[rows.original_split == split][["dataset", "signer_id"]].astype(str).values))
        for split in DEVELOPMENT_SPLITS
    }
    if signers["train"] & signers["validation"]:
        raise ValueError("Signer leakage remains between train and validation.")
    hashes = {
        split: {sha256(Path(path)) for path in rows.loc[rows.original_split == split, "video_path_abs"]}
        for split in DEVELOPMENT_SPLITS
    }
    if hashes["train"] & hashes["validation"]:
        raise ValueError("Raw content leakage remains between train and validation.")


class SequenceDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, mapping: dict[str, int], normalizer: FeatureNormalizer):
        self.rows = list(rows.itertuples(index=False))
        self.mapping = mapping
        self.normalizer = normalizer

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        with np.load(row.feature_path, allow_pickle=False) as data:
            features = np.asarray(data["lstm"][:, :177], dtype=np.float32)
            presence = np.asarray(data["presence"], dtype=np.float32)
        sequence = self.normalizer.transform_lstm(features, presence)
        label = self.mapping[str(row.gloss_meta)]
        return torch.from_numpy(sequence.copy()), torch.tensor(label, dtype=torch.long)


def atomic_torch_save(obj: object, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(obj, temporary)
    temporary.replace(destination)


def collect_sequences(rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    features, presence = [], []
    for row in rows.itertuples(index=False):
        with np.load(row.feature_path, allow_pickle=False) as data:
            features.append(np.asarray(data["lstm"][:, :177], dtype=np.float32))
            presence.append(np.asarray(data["presence"], dtype=np.float32))
    return np.asarray(features), np.asarray(presence)


def evaluate(model: nn.Module, loader: DataLoader) -> tuple[float, list[int], list[int]]:
    model.eval()
    loss_sum = 0.0
    labels, predictions = [], []
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for features, target in loader:
            logits = model(features)
            loss_sum += criterion(logits, target).item() * len(target)
            predictions.extend(logits.argmax(1).tolist())
            labels.extend(target.tolist())
    return loss_sum / len(labels), labels, predictions


def save_curves(history: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.epoch, history.train_loss, label="train")
    axes[0].plot(history.epoch, history.validation_loss, label="validation")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(history.epoch, history.train_accuracy, label="train accuracy")
    axes[1].plot(history.epoch, history.validation_accuracy, label="validation accuracy")
    axes[1].plot(history.epoch, history.validation_macro_f1, label="validation macro-F1")
    axes[1].set_title("Validation metrics")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(OUT / "training_curves.png", dpi=150)
    plt.close(figure)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    mapping = json.loads((META / "label_mapping.json").read_text(encoding="utf-8"))
    all_rows = load_development_rows()
    effective, exclusions = content_safe_view(all_rows)
    validate_development_view(effective, mapping)
    exclusions.to_csv(OUT / "content_exclusions.csv", index=False)
    train_rows = effective[effective.original_split == "train"]
    val_rows = effective[effective.original_split == "validation"]
    train_features, train_presence = collect_sequences(train_rows)
    normalizer = FeatureNormalizer.fit(train_features, train_presence)
    (OUT / "normalization.json").write_text(json.dumps(normalizer.to_dict(), indent=2), encoding="utf-8")

    train_labels = train_rows.gloss_meta.astype(str).map(mapping).to_numpy()
    class_counts = np.bincount(train_labels, minlength=100)
    class_weights = len(train_labels) / (100 * np.maximum(class_counts, 1))
    pd.DataFrame({"label": range(100), "count": class_counts, "weight": class_weights}).to_csv(
        OUT / "training_class_counts.csv", index=False
    )
    train_loader = DataLoader(
        SequenceDataset(train_rows, mapping, normalizer), batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        SequenceDataset(val_rows, mapping, normalizer), batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    model = LSTMClassifier(100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights.astype(np.float32)))
    history = []
    best_f1, stale = -1.0, 0
    started = time.perf_counter()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        loss_sum = correct = total = 0
        for features, target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += loss.item() * len(target)
            correct += (logits.argmax(1) == target).sum().item()
            total += len(target)
        train_loss, train_accuracy = loss_sum / total, correct / total
        validation_loss, ys, predictions = evaluate(model, val_loader)
        precision, recall, f1, _ = precision_recall_fscore_support(
            ys, predictions, labels=list(range(100)), zero_division=0
        )
        validation_accuracy = accuracy_score(ys, predictions)
        validation_macro_f1 = float(f1.mean())
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
                "validation_macro_precision": float(precision.mean()),
                "validation_macro_recall": float(recall.mean()),
                "validation_macro_f1": validation_macro_f1,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        pd.DataFrame(history).to_csv(OUT / "training_history.csv", index=False)
        if validation_macro_f1 > best_f1:
            best_f1, stale = validation_macro_f1, 0
            atomic_torch_save(
                {
                    "model_state_dict": model.state_dict(),
                    "label_mapping": mapping,
                    "input_shape": [30, 180],
                    "feature_key": "lstm",
                    "normalization": normalizer.to_dict(),
                    "best_validation_macro_f1": best_f1,
                    "epoch": epoch,
                    "seed": SEED,
                },
                OUT / "best_lstm.pt",
            )
        else:
            stale += 1
        print(
            f"Epoch {epoch:03d}/{MAX_EPOCHS}: train_loss={train_loss:.4f} "
            f"val_loss={validation_loss:.4f} val_macro_f1={validation_macro_f1:.4f}",
            flush=True,
        )
        if stale >= PATIENCE:
            break

    history_frame = pd.DataFrame(history)
    save_curves(history_frame)
    checkpoint = torch.load(OUT / "best_lstm.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    validation_loss, ys, predictions = evaluate(model, val_loader)
    matrix = confusion_matrix(ys, predictions, labels=list(range(100)))
    np.savetxt(OUT / "validation_confusion_matrix.csv", matrix, delimiter=",", fmt="%d")
    report = classification_report(
        ys,
        predictions,
        labels=list(range(100)),
        target_names=[g for g, _ in sorted(mapping.items(), key=lambda item: item[1])],
        zero_division=0,
        output_dict=True,
    )
    (OUT / "validation_per_class_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    best_row = history_frame.loc[history_frame.validation_macro_f1.idxmax()]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        "\n".join(
            [
                "# Controlled YOLO11 + MediaPipe Hands LSTM v2",
                "",
                "This experiment uses only train and validation splits. The held-out test split was not loaded or evaluated.",
                "",
                f"- Effective train/validation rows: {len(train_rows)} / {len(val_rows)}",
                f"- Training duration: {time.perf_counter() - started:.1f} seconds",
                f"- Best epoch: {int(best_row.epoch)}",
                f"- Best validation macro-F1: {best_row.validation_macro_f1:.6f}",
                f"- Best validation accuracy: {best_row.validation_accuracy:.6f}",
                f"- Normalization: `{OUT.relative_to(ROOT) / 'normalization.json'}`",
                f"- Checkpoint: `{OUT.relative_to(ROOT) / 'best_lstm.pt'}`",
                "",
                "Class weighting uses inverse-frequency weighted cross-entropy; no sampler or duplicated samples was used.",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
