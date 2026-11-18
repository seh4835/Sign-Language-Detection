"""Extract a bounded YOLO11 + MediaPipe Hands dataset sample."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.keypoints.yolo_pose_hands import YoloPoseHandsExtractor, preprocess_features


FRAME_COUNT = 30
FEATURE_DIM = 177
LSTM_DIM = 180
SPLIT_NAMES = ("train", "validation", "test")
DEFAULT_MODEL = Path(r"outputs\yolo11_pose_test\yolo11n-pose.pt")
DEFAULT_OUTPUT = Path(r"data\keypoints_yolo_pose_hands")
DEFAULT_REPORT = Path(r"reports\yolo11_migration\extraction_smoke_test.md")
SUCCESS_COLUMNS = (
    "video_id", "dataset", "signer_id", "gloss", "canonical_label", "original_split",
    "video_path", "feature_file", "frames", "body_detection_rate",
    "left_hand_detection_rate", "right_hand_detection_rate", "processing_seconds",
)
FAILURE_COLUMNS = (
    "video_id", "dataset", "signer_id", "gloss", "canonical_label", "original_split",
    "video_path", "error",
)
INVALID_COLUMNS = ("original_split", "video_id", "feature_file", "error")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate existing NPZs and recover success metadata without extracting videos.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show resume decisions without loading models or extracting videos.",
    )
    parser.add_argument(
        "--retry-safe-sources",
        action="store_true",
        help=(
            "Retry only previously failed rows whose recorded error was a raw-video "
            "match-count issue; short/decode failures are left untouched."
        ),
    )
    return parser.parse_args()


def load_manifest(root: Path) -> list[dict[str, str]]:
    metadata = root / "data" / "metadata"
    rows: list[dict[str, str]] = []
    for split in SPLIT_NAMES:
        path = metadata / f"{split}_signer_disjoint.csv"
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                row["original_split"] = split
                rows.append(row)
    return rows


def select_rows(rows: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
    by_split = {split: [row for row in rows if row["original_split"] == split] for split in SPLIT_NAMES}
    if limit == 20:
        counts = {"train": 7, "validation": 7, "test": 6}
    elif limit == sum(len(items) for items in by_split.values()):
        counts = {split: len(items) for split, items in by_split.items()}
    else:
        raise ValueError("Use --limit 20 for the smoke test or the full canonical dataset size.")
    return [row for split in SPLIT_NAMES for row in by_split[split][: counts[split]]]


def index_videos(root: Path, video_ids: set[str]) -> dict[str, list[Path]]:
    """Match raw filenames against canonical IDs without stripping ID characters.

    Parsing at the final ``__`` loses the leading underscore when a video ID
    itself starts with ``_`` because the filename contains three underscores.
    Matching the exact canonical-ID suffix keeps that character intact.
    """
    index: dict[str, list[Path]] = defaultdict(list)
    suffixes = {video_id: f"__{video_id}" for video_id in video_ids}
    for path in (root / "data" / "raw").rglob("*"):
        if path.is_file() and path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
            for video_id, suffix in suffixes.items():
                if path.stem.endswith(suffix):
                    prefix = path.stem[: -len(suffix)]
                    # If the remaining prefix ends in ``_``, that character
                    # belongs to a leading-underscore ID and this is a
                    # shorter false suffix match (e.g. ``ID`` vs ``_ID``).
                    if not prefix.endswith("_"):
                        index[video_id].append(path)
    return index


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_video_candidates(candidates: list[Path]) -> tuple[Path | None, str, str | None]:
    """Resolve aliases only when their contents are byte-identical."""
    if not candidates:
        return None, "No raw video matches the canonical video ID.", None
    hashes: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(set(candidates), key=lambda item: str(item).casefold()):
        hashes[sha256_file(path)].append(path)
    if len(hashes) != 1:
        paths = "; ".join(str(path) for path in sorted(set(candidates), key=lambda item: str(item).casefold()))
        return None, f"Multiple raw videos have different content ({len(hashes)} SHA-256 groups): {paths}", None
    digest, identical_paths = next(iter(hashes.items()))
    return identical_paths[0], "", digest


def index_successful_source_hashes(
    root: Path,
    successes: dict[tuple[str, str, str], dict[str, str]],
) -> dict[str, set[tuple[str, str, str]]]:
    """Index raw source content already represented by valid feature files."""
    hashes: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    path_hashes: dict[Path, str] = {}
    for key, row in successes.items():
        source_value = row.get("video_path", "")
        if not source_value:
            continue
        source_path = Path(source_value)
        if not source_path.is_absolute():
            source_path = root / source_path
        try:
            source_path = source_path.resolve(strict=True)
        except OSError:
            continue
        digest = path_hashes.get(source_path)
        if digest is None:
            digest = sha256_file(source_path)
            path_hashes[source_path] = digest
        hashes[digest].add(key)
    return hashes


def source_content_conflicts(
    source_hashes: dict[str, set[tuple[str, str, str]]],
    digest: str,
    key: tuple[str, str, str],
) -> list[tuple[str, str, str]]:
    """Return other canonical rows already using the same video bytes."""
    return sorted(source_hashes.get(digest, set()) - {key})


def sample_indices(total_frames: int) -> np.ndarray:
    if total_frames < FRAME_COUNT:
        raise RuntimeError(f"Video has only {total_frames} frames; need at least {FRAME_COUNT}.")
    return np.linspace(0, total_frames - 1, FRAME_COUNT, dtype=np.int64)


def output_path(output_dir: Path, row: dict[str, str]) -> Path:
    safe_name = f"{row['dataset']}__{row['video_id']}.npz"
    return output_dir / row["original_split"] / safe_name


def inspect_output(path: Path) -> tuple[bool, str, np.ndarray | None]:
    if not path.is_file():
        return False, "file does not exist", None
    try:
        with np.load(path, allow_pickle=False) as data:
            expected = {"raw": (FRAME_COUNT, FEATURE_DIM), "presence": (FRAME_COUNT, 3), "lstm": (FRAME_COUNT, LSTM_DIM)}
            arrays = {name: data[name] for name in expected}
            for name, shape in expected.items():
                if arrays[name].shape != shape:
                    return False, f"{name} has shape {arrays[name].shape}; expected {shape}", None
                if not np.isfinite(arrays[name]).all():
                    return False, f"{name} contains non-finite values", None
            return True, "", np.asarray(arrays["presence"], dtype=np.float64)
    except Exception as error:
        return False, f"could not read NPZ: {error}", None


def valid_output(path: Path) -> bool:
    return inspect_output(path)[0]


def atomic_write_csv(path: Path, rows: list[dict[str, str]], columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows({column: row.get(column, "") for column in columns} for row in rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def save_npz_without_overwrite(path: Path, **arrays: np.ndarray) -> None:
    """Publish a complete new NPZ atomically; fail if the destination exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".npz.tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        # A same-directory hard link publishes atomically and cannot replace an existing file.
        os.link(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def row_key(row: dict[str, str]) -> tuple[str, str, str]:
    return row["original_split"], row["dataset"], row["video_id"]


def is_source_resolution_failure(error: str) -> bool:
    return error.startswith("Expected one raw video, found ")


def recover_existing_outputs(
    root: Path,
    output_dir: Path,
    rows: list[dict[str, str]],
    mapping: dict[str, int],
    video_index: dict[str, list[Path]],
    prior_successes: list[dict[str, str]],
) -> tuple[dict[tuple[str, str, str], dict[str, str]], list[dict[str, str]]]:
    manifest = {row_key(row): row for row in rows}
    previous = {
        (row.get("original_split", ""), row.get("dataset", ""), row.get("video_id", "")): row
        for row in prior_successes
    }
    successes: dict[tuple[str, str, str], dict[str, str]] = {}
    invalid: list[dict[str, str]] = []
    for feature_path in sorted(output_dir.rglob("*.npz")):
        split = feature_path.parent.name
        if "__" not in feature_path.stem:
            invalid.append({"original_split": split, "video_id": "", "feature_file": str(feature_path), "error": "Filename is not dataset__video_id."})
            continue
        dataset, video_id = feature_path.stem.split("__", 1)
        key = (split, dataset, video_id)
        row = manifest.get(key)
        if row is None:
            invalid.append({"original_split": split, "video_id": video_id, "feature_file": str(feature_path), "error": "No matching canonical manifest row."})
            continue
        valid, error, presence = inspect_output(feature_path)
        if not valid or presence is None:
            invalid.append({"original_split": split, "video_id": video_id, "feature_file": str(feature_path), "error": error})
            continue
        old = previous.get(key, {})
        matches = video_index.get(video_id, [])
        video_path = old.get("video_path", "")
        if not video_path:
            resolved_path, _, _ = resolve_video_candidates(matches)
            if resolved_path is not None:
                video_path = str(resolved_path.relative_to(root))
        presence_rates = presence.mean(axis=0)
        successes[key] = {
            "video_id": video_id,
            "dataset": dataset,
            "signer_id": row["signer_id"],
            "gloss": row["gloss"],
            "canonical_label": str(mapping[row["gloss"]]),
            "original_split": split,
            "video_path": video_path,
            "feature_file": str(feature_path),
            "frames": str(FRAME_COUNT),
            "body_detection_rate": f"{presence_rates[0]:.6f}",
            "left_hand_detection_rate": f"{presence_rates[1]:.6f}",
            "right_hand_detection_rate": f"{presence_rates[2]:.6f}",
            # The original runtime for recovered outputs is not encoded in the NPZ.
            "processing_seconds": old.get("processing_seconds", ""),
        }
    return successes, invalid


def extract_row(
    row: dict[str, str],
    video_path: Path,
    feature_path: Path,
    model: Any,
    root: Path,
) -> dict[str, str]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    extractor = YoloPoseHandsExtractor(model)
    raw_frames: list[np.ndarray] = []
    processed_frames: list[np.ndarray] = []
    presence_frames: list[np.ndarray] = []
    started = time.perf_counter()
    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        for frame_index in sample_indices(total_frames):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            success, frame = capture.read()
            if not success:
                raise RuntimeError(f"Could not read sampled frame {frame_index}.")
            raw = extractor.process_frame(frame)
            processed, indicators = preprocess_features(
                raw, width, height, extractor.last_presence
            )
            raw_frames.append(raw)
            processed_frames.append(processed)
            presence_frames.append(indicators)
    finally:
        capture.release()
        extractor.close()

    raw_array = np.asarray(raw_frames, dtype=np.float32)
    presence_array = np.asarray(presence_frames, dtype=np.float32)
    normalized_array = np.asarray(processed_frames, dtype=np.float32)
    lstm_array = np.concatenate((normalized_array, presence_array), axis=1)
    if (
        raw_array.shape != (FRAME_COUNT, FEATURE_DIM)
        or presence_array.shape != (FRAME_COUNT, 3)
        or lstm_array.shape != (FRAME_COUNT, LSTM_DIM)
        or not np.isfinite(lstm_array).all()
    ):
        raise ValueError("Extraction produced invalid output arrays.")
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    save_npz_without_overwrite(feature_path, raw=raw_array, presence=presence_array, lstm=lstm_array)
    presence_rates = presence_array.mean(axis=0)
    return {
        "video_id": row["video_id"],
        "dataset": row["dataset"],
        "signer_id": row["signer_id"],
        "gloss": row["gloss"],
        "canonical_label": str(row["_canonical_label"]),
        "original_split": row["original_split"],
        "video_path": str(video_path.relative_to(root)),
        "feature_file": str(feature_path),
        "frames": str(FRAME_COUNT),
        "body_detection_rate": f"{presence_rates[0]:.6f}",
        "left_hand_detection_rate": f"{presence_rates[1]:.6f}",
        "right_hand_detection_rate": f"{presence_rates[2]:.6f}",
        "processing_seconds": f"{time.perf_counter() - started:.4f}",
    }


def write_csv(path: Path, rows: list[dict[str, str]], columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({column: row.get(column, "") for column in columns} for row in rows)


def write_report(
    path: Path,
    successes: list[dict[str, str]],
    failures: list[dict[str, str]],
    elapsed: float,
) -> None:
    split_counts = {split: sum(row["original_split"] == split for row in successes) for split in SPLIT_NAMES}
    lines = [
        "# YOLO11 + MediaPipe Hands Extraction Smoke Test",
        "",
        "Bounded first-run extraction only; no full-dataset extraction or training was performed.",
        "",
        f"- Selected videos: {len(successes) + len(failures)}",
        f"- Successful: {len(successes)}",
        f"- Failed: {len(failures)}",
        f"- Successful split representation: train={split_counts['train']}, validation={split_counts['validation']}, test={split_counts['test']}",
        f"- Total wall-clock time: {elapsed:.2f} seconds",
        "- Sampling: 30 uniformly spaced frame indices from each complete video",
        "- Raw shape: `(30,177)`",
        "- Presence shape: `(30,3)`",
        "- LSTM shape: `(30,180)`",
        "",
        "## Outputs",
        "",
        "Each successful `.npz` contains `raw`, `presence`, and `lstm` arrays. "
        "Success and failure metadata are stored in the extraction output directory.",
        "",
        "## Failures",
        "",
    ]
    if failures:
        lines.extend(f"- `{row['video_id']}` ({row['original_split']}): {row['error']}" for row in failures)
    else:
        lines.append("None.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if args.validate_only and args.dry_run:
        raise ValueError("Choose either --validate-only or --dry-run, not both.")
    manifest_rows = load_manifest(root)
    rows = select_rows(manifest_rows, args.limit)
    mapping = json.loads((root / "data" / "metadata" / "label_mapping.json").read_text(encoding="utf-8"))
    for row in manifest_rows:
        if row["gloss"] not in mapping:
            raise KeyError(f"Gloss {row['gloss']!r} is missing from label_mapping.json.")
    for row in manifest_rows:
        row["_canonical_label"] = str(mapping[row["gloss"]])
    video_ids = {row["video_id"] for row in manifest_rows}
    video_index = index_videos(root, video_ids)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    success_path = args.output_dir / "extraction_metadata.csv"
    failure_path = args.output_dir / "extraction_failures.csv"
    invalid_path = args.output_dir / "extraction_invalid_outputs.csv"
    existing_successes: list[dict[str, str]] = []
    if success_path.is_file():
        with success_path.open(newline="", encoding="utf-8") as handle:
            existing_successes = list(csv.DictReader(handle))
    existing_failures: list[dict[str, str]] = []
    if failure_path.is_file():
        with failure_path.open(newline="", encoding="utf-8") as handle:
            existing_failures = list(csv.DictReader(handle))
    if args.retry_safe_sources:
        prior_failure_keys = {
            (row.get("original_split", ""), row.get("dataset", ""), row.get("video_id", "")): row
            for row in existing_failures
        }
        rows = [
            row
            for row in rows
            if row_key(row) in prior_failure_keys
            and is_source_resolution_failure(prior_failure_keys[row_key(row)].get("error", ""))
        ]

    successes, invalid = recover_existing_outputs(
        root, args.output_dir, manifest_rows, mapping, video_index, existing_successes
    )
    source_hashes = index_successful_source_hashes(root, successes)
    failures_by_key = {
        (row.get("original_split", ""), row.get("dataset", ""), row.get("video_id", "")): row
        for row in existing_failures
        if (row.get("original_split", ""), row.get("dataset", ""), row.get("video_id", "")) not in successes
    }
    if args.validate_only:
        atomic_write_csv(success_path, list(successes.values()), SUCCESS_COLUMNS)
        atomic_write_csv(failure_path, list(failures_by_key.values()), FAILURE_COLUMNS)
        atomic_write_csv(invalid_path, invalid, INVALID_COLUMNS)
        print(f"validated={len(successes) + len(invalid)} valid={len(successes)} invalid_or_unmatched={len(invalid)}")
        print(f"metadata_rows={len(successes)} failure_rows_preserved={len(failures_by_key)}")
        return 0 if not invalid else 1

    if args.dry_run:
        skipped = pending = blocked = 0
        for batch_start in range(0, len(rows), args.batch_size):
            batch = rows[batch_start : batch_start + args.batch_size]
            batch_skipped = batch_pending = batch_blocked = 0
            for offset, row in enumerate(batch):
                index = batch_start + offset + 1
                feature_path = output_path(args.output_dir, row)
                key = row_key(row)
                if key in successes:
                    status = "SUCCESS (skip valid existing NPZ)"
                    batch_skipped += 1
                elif feature_path.exists():
                    status = "FAILURE (invalid existing NPZ preserved)"
                    batch_blocked += 1
                else:
                    matches = video_index.get(row["video_id"], [])
                    resolved, error, digest = resolve_video_candidates(matches)
                    if resolved is None or digest is None:
                        status = f"BLOCKED ({error})"
                        batch_blocked += 1
                    else:
                        duplicates = source_content_conflicts(source_hashes, digest, key)
                        if duplicates:
                            status = f"BLOCKED (source content already used by {sorted(duplicates)})"
                            batch_blocked += 1
                        else:
                            status = "PENDING (would extract)"
                            batch_pending += 1
                print(f"[{index}/{len(rows)}] {row['video_id']} {row['original_split']} {status} elapsed=0.00s")
            skipped += batch_skipped
            pending += batch_pending
            blocked += batch_blocked
            print(
                f"batch {batch_start // args.batch_size + 1}: skipped={batch_skipped} "
                f"pending={batch_pending} blocked={batch_blocked}"
            )
        print(f"dry_run_total={len(rows)} skipped={skipped} pending={pending} blocked={blocked}; no extraction performed")
        return 0

    if any(not output_path(args.output_dir, row).exists() and row_key(row) not in successes for row in rows):
        if not args.model.is_file():
            raise FileNotFoundError(f"Model does not exist: {args.model}")

    # Persist reconstructed records before starting so an interruption cannot leave stale metadata.
    failures_by_key = {key: value for key, value in failures_by_key.items() if key not in successes}
    atomic_write_csv(success_path, list(successes.values()), SUCCESS_COLUMNS)
    atomic_write_csv(failure_path, list(failures_by_key.values()), FAILURE_COLUMNS)
    atomic_write_csv(invalid_path, invalid, INVALID_COLUMNS)
    model = None
    started = time.perf_counter()
    successes_in_run = failures_in_run = 0
    try:
        for batch_start in range(0, len(rows), args.batch_size):
            batch = rows[batch_start : batch_start + args.batch_size]
            batch_successes = batch_failures = 0
            for offset, row in enumerate(batch):
                index = batch_start + offset + 1
                video_started = time.perf_counter()
                key = row_key(row)
                feature_path = output_path(args.output_dir, row)
                status = ""
                if key in successes:
                    status = "SUCCESS (skipped valid existing NPZ)"
                    batch_successes += 1
                elif feature_path.exists():
                    status = "FAILURE (invalid existing NPZ preserved)"
                    batch_failures += 1
                else:
                    matches = video_index.get(row["video_id"], [])
                    video_path, error, source_digest = resolve_video_candidates(matches)
                    if video_path is None or source_digest is None:
                        failure_row = {
                            **row,
                            "canonical_label": row["_canonical_label"],
                            "video_path": "",
                            "error": error,
                        }
                        failures_by_key[key] = failure_row
                        status = f"FAILURE ({error})"
                        batch_failures += 1
                    else:
                        duplicate_keys = source_content_conflicts(source_hashes, source_digest, key)
                        if duplicate_keys:
                            error = (
                                "Raw video content is already assigned to canonical row(s): "
                                + ", ".join(str(item) for item in sorted(duplicate_keys))
                            )
                            failures_by_key[key] = {
                                **row,
                                "canonical_label": row["_canonical_label"],
                                "video_path": str(video_path.relative_to(root)),
                                "error": error,
                            }
                            status = f"FAILURE ({error})"
                            batch_failures += 1
                        else:
                            try:
                                if model is None:
                                    from ultralytics import YOLO
                                    model = YOLO(str(args.model))
                                result = extract_row(row, video_path, feature_path, model, root)
                                successes[key] = result
                                source_hashes[source_digest].add(key)
                                failures_by_key.pop(key, None)
                                status = "SUCCESS"
                                batch_successes += 1
                            except FileExistsError:
                                # A file appeared after the scan; never overwrite it. Validate on the next run.
                                error = "Output appeared during extraction; preserved without overwrite."
                                failures_by_key[key] = {
                                    **row,
                                    "canonical_label": row["_canonical_label"],
                                    "video_path": str(video_path.relative_to(root)),
                                    "error": error,
                                }
                                status = f"FAILURE ({error})"
                                batch_failures += 1
                            except Exception as error:
                                failures_by_key[key] = {
                                    **row,
                                    "canonical_label": row["_canonical_label"],
                                    "video_path": str(video_path.relative_to(root)),
                                    "error": str(error),
                                }
                                status = f"FAILURE ({error})"
                                batch_failures += 1
                print(
                    f"[{index}/{len(rows)}] {row['video_id']} {row['original_split']} "
                    f"{status} elapsed={time.perf_counter() - video_started:.2f}s"
                )
            successes_in_run += batch_successes
            failures_in_run += batch_failures
            atomic_write_csv(success_path, list(successes.values()), SUCCESS_COLUMNS)
            atomic_write_csv(failure_path, list(failures_by_key.values()), FAILURE_COLUMNS)
            atomic_write_csv(invalid_path, invalid, INVALID_COLUMNS)
            print(
                f"batch {batch_start // args.batch_size + 1}: "
                f"success={batch_successes} failure={batch_failures} "
                f"checkpointed metadata ({len(successes)} valid outputs total)"
            )
    except KeyboardInterrupt:
        atomic_write_csv(success_path, list(successes.values()), SUCCESS_COLUMNS)
        atomic_write_csv(failure_path, list(failures_by_key.values()), FAILURE_COLUMNS)
        atomic_write_csv(invalid_path, invalid, INVALID_COLUMNS)
        print("Interrupted safely; completed outputs and current metadata checkpoint were preserved.")
        return 130
    print(
        f"run_complete success={successes_in_run} failure={failures_in_run} "
        f"valid_outputs_total={len(successes)} elapsed={time.perf_counter() - started:.2f}s"
    )
    return 0 if failures_in_run == 0 and not invalid else 1


if __name__ == "__main__":
    raise SystemExit(main())
