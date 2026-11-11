"""Train and evaluate the existing pose-and-hands Transformer over five folds."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, top_k_accuracy_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.cross_validation import (
    N_FOLDS,
    RANDOM_STATE,
    create_fold_assignments,
    fit_normalization,
    fold_indices,
    load_development_data,
)
from src.models.transformer_pose_hands import (
    MODEL_FEATURE_DIM,
    TransformerClassifier,
    load_label_mapping,
)


ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "data" / "metadata"
REPORTS = ROOT / "reports" / "cross_validation"
CHECKPOINTS = REPORTS / "checkpoints"
DEVICE = torch.device("cpu")


class SequenceDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


def load_features(dataframe: pd.DataFrame, mapping: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    features = []
    labels = []
    for _, row in dataframe.iterrows():
        path = Path(str(row["keypoint_path"]))
        if not path.is_absolute():
            path = ROOT / path
        array = np.load(path, allow_pickle=False)
        if array.ndim != 2 or array.shape[1] != 1662:
            raise ValueError(f"Expected (frames, 1662) at {path}, got {array.shape}")
        features.append(array[:, 1404:1662].astype(np.float32))
        labels.append(mapping[str(row["gloss"])])
    return np.stack(features), np.asarray(labels, dtype=np.int64)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(
    model: nn.Module, loader: DataLoader, labels: np.ndarray, class_count: int
) -> tuple[float, float, float, np.ndarray]:
    model.eval()
    logits_list = []
    with torch.no_grad():
        for features, _ in loader:
            logits_list.append(model(features.to(DEVICE)))
    logits = torch.cat(logits_list)
    probabilities = logits.softmax(dim=1).cpu().numpy()
    predictions = probabilities.argmax(axis=1)
    top1 = float((predictions == labels).mean())
    top5 = float(
        top_k_accuracy_score(
            labels, probabilities, k=min(5, class_count),
            labels=np.arange(class_count),
        )
    )
    loss = float(nn.CrossEntropyLoss()(logits, torch.tensor(labels)).item())
    return loss, top1, top5, probabilities


def train_fold(
    fold: int,
    assignments: pd.DataFrame,
    mapping: dict[str, int],
    *,
    epochs: int,
    patience: int,
    batch_size: int,
) -> tuple[dict[str, float | int], pd.DataFrame, pd.DataFrame]:
    train_indices, validation_indices = fold_indices(assignments, fold)
    train_df = assignments.iloc[train_indices]
    validation_df = assignments.iloc[validation_indices]
    raw_train, y_train = load_features(train_df, mapping)
    raw_validation, y_validation = load_features(validation_df, mapping)
    mean, std = fit_normalization(raw_train)
    train_features = (raw_train - mean) / std
    validation_features = (raw_validation - mean) / std

    train_loader = DataLoader(
        SequenceDataset(train_features, y_train), batch_size=batch_size, shuffle=True
    )
    validation_loader = DataLoader(
        SequenceDataset(validation_features, y_validation), batch_size=64, shuffle=False
    )
    model = TransformerClassifier(
        input_dim=MODEL_FEATURE_DIM, classes=len(mapping)
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    criterion = nn.CrossEntropyLoss()
    best_loss = float("inf")
    best_state = None
    stale = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        count = 0
        for features, labels in train_loader:
            optimizer.zero_grad()
            logits = model(features.to(DEVICE))
            loss = criterion(logits, labels.to(DEVICE))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += int((logits.argmax(1).cpu() == labels).sum())
            count += len(labels)

        validation_loss, validation_top1, validation_top5, _ = evaluate(
            model, validation_loader, y_validation, len(mapping)
        )
        scheduler.step(validation_loss)
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / count,
                "train_accuracy": correct / count,
                "validation_loss": validation_loss,
                "validation_top1_accuracy": validation_top1,
                "validation_top5_accuracy": validation_top5,
            }
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break

    if best_state is None:
        raise RuntimeError(f"Fold {fold} did not produce a checkpoint.")
    model.load_state_dict(best_state)
    final_loss, top1, top5, probabilities = evaluate(
        model, validation_loader, y_validation, len(mapping)
    )
    names = [label for label, _ in sorted(mapping.items(), key=lambda pair: pair[1])]
    per_class = pd.DataFrame(
        classification_report(
            y_validation,
            probabilities.argmax(axis=1),
            labels=np.arange(len(mapping)),
            target_names=names,
            output_dict=True,
            zero_division=0,
        )
    ).transpose().reset_index(names="class")
    checkpoint_path = CHECKPOINTS / f"transformer_pose_hands_fold_{fold}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": MODEL_FEATURE_DIM,
            "dim": 128,
            "heads": 4,
            "layers": 2,
            "classes": len(mapping),
            "mean": mean,
            "std": std,
            "fold": fold,
            "development_only": True,
        },
        checkpoint_path,
    )
    result = {
        "fold": fold,
        "epochs_trained": len(history),
        "best_validation_loss": best_loss,
        "validation_loss": final_loss,
        "top1_accuracy": top1,
        "top5_accuracy": top5,
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
    }
    return result, pd.DataFrame(history), per_class


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    REPORTS.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    mapping = load_label_mapping(META / "label_mapping.json")
    development = load_development_data(
        META / "train_signer_disjoint.csv",
        META / "validation_signer_disjoint.csv",
    )
    assignments, fold_report = create_fold_assignments(
        development, n_splits=N_FOLDS, random_state=RANDOM_STATE
    )
    assignments.to_csv(REPORTS / "fold_assignments.csv", index=False)
    (REPORTS / "fold_validation.json").write_text(
        json.dumps(
            {"n_folds": N_FOLDS, "random_state": RANDOM_STATE, "folds": fold_report},
            indent=2,
        ),
        encoding="utf-8",
    )
    class_rows = []
    for item in fold_report:
        for partition in ("training", "validation"):
            for label, count in item[f"{partition}_class_counts"].items():
                class_rows.append(
                    {"fold": item["fold"], "partition": partition, "gloss": label, "count": count}
                )
    pd.DataFrame(class_rows).to_csv(REPORTS / "class_counts.csv", index=False)
    limitations = {
        str(item["fold"]): item["missing_validation_classes"]
        for item in fold_report
        if item["missing_validation_classes"]
    }
    (REPORTS / "coverage_limitations.json").write_text(
        json.dumps({"missing_validation_classes_by_fold": limitations}, indent=2),
        encoding="utf-8",
    )
    if limitations:
        print("CLASS-COVERAGE LIMITATIONS (validation):")
        for fold, missing in limitations.items():
            print(f"  Fold {fold}: {', '.join(missing)}")
    else:
        print("All folds contain all development classes in validation.")
    if args.validate_only:
        return

    results = []
    per_class = []
    for fold in range(1, N_FOLDS + 1):
        set_seed(RANDOM_STATE + fold)
        result, history, metrics = train_fold(
            fold, assignments, mapping, epochs=args.epochs,
            patience=args.patience, batch_size=args.batch_size
        )
        history.to_csv(REPORTS / f"fold_{fold}_history.csv", index=False)
        metrics.insert(0, "fold", fold)
        metrics.to_csv(REPORTS / f"fold_{fold}_per_class_metrics.csv", index=False)
        results.append(result)
        per_class.append(metrics)
        print(
            f"Fold {fold}: Top-1={result['top1_accuracy']:.4f}, "
            f"Top-5={result['top5_accuracy']:.4f}"
        )
    fold_summary = pd.DataFrame(results)
    summary = fold_summary.copy()
    summary.loc[len(summary)] = {
        "fold": "mean",
        "top1_accuracy": fold_summary["top1_accuracy"].mean(),
        "top5_accuracy": fold_summary["top5_accuracy"].mean(),
    }
    summary.loc[len(summary)] = {
        "fold": "std",
        "top1_accuracy": fold_summary["top1_accuracy"].std(ddof=1),
        "top5_accuracy": fold_summary["top5_accuracy"].std(ddof=1),
    }
    summary.to_csv(REPORTS / "summary.csv", index=False)
    pd.concat(per_class, ignore_index=True).to_csv(
        REPORTS / "per_class_metrics.csv", index=False
    )


if __name__ == "__main__":
    main()
