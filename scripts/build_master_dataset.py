import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
KEYPOINT_DIR = PROJECT_ROOT / "data" / "keypoints"

OUTPUT_FILE = METADATA_DIR / "master_dataset_review.csv"
REPORT_FILE = METADATA_DIR / "master_dataset_review_report.txt"

MANIFEST_FILES = [
    METADATA_DIR / "combined_expanded_manifest.csv",
    METADATA_DIR / "combined_download_manifest.csv",
    METADATA_DIR / "balanced_batch_02_manifest.csv",
    METADATA_DIR / "remaining_manifest.csv",
    METADATA_DIR / "recovery_manifest.csv",
]

KEYPOINT_DIRS = [
    KEYPOINT_DIR / "wlasl",
    KEYPOINT_DIR / "expanded_batch_01",
    KEYPOINT_DIR / "balanced_batch_02",
    KEYPOINT_DIR / "remaining_downloads",
    KEYPOINT_DIR / "recovery",
]

EXCLUDED_VIDEO_IDS = {"N5ZUQXL4"}
EXPECTED_SHAPE = (30, 1662)

FIELDS = [
    "dataset", "split", "gloss", "label", "signer_id",
    "start", "end", "video_id", "keypoint_path",
]


def normalize(value):
    return str(value or "").strip().lower()


def get_file_key(path):
    # Filenames: dataset__gloss__video_id.npy
    # or dataset__gloss__split__video_id.npy
    parts = path.stem.split("__")
    if len(parts) >= 4:
        return normalize(parts[0]), parts[-1].strip()
    if len(parts) == 3:
        return normalize(parts[0]), parts[2].strip()
    return None


def main():
    metadata_by_key = {}
    duplicate_metadata = []

    for manifest_path in MANIFEST_FILES:
        if not manifest_path.is_file():
            print(f"Manifest not found (skipping): {manifest_path.name}")
            continue

        with open(manifest_path, "r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                dataset = normalize(row.get("dataset"))
                video_id = str(row.get("video_id", "")).strip()

                if not dataset or not video_id:
                    continue

                key = (dataset, video_id)
                if key in metadata_by_key:
                    # Keep the first source deterministically.
                    previous = metadata_by_key[key]
                    if normalize(previous.get("gloss")) != normalize(row.get("gloss")):
                        duplicate_metadata.append(
                            f"{dataset}/{video_id}: conflicting gloss metadata"
                        )
                    continue

                metadata_by_key[key] = row

    records_by_key = {}
    invalid_files = []
    missing_metadata = []

    for directory in KEYPOINT_DIRS:
        if not directory.is_dir():
            print(f"Keypoint directory not found (skipping): {directory.name}")
            continue

        for path in sorted(directory.glob("*.npy")):
            key = get_file_key(path)
            if not key:
                invalid_files.append(f"{path.name}: unrecognized filename")
                continue

            dataset, video_id = key
            if video_id in EXCLUDED_VIDEO_IDS:
                continue

            metadata = metadata_by_key.get(key)
            if metadata is None:
                missing_metadata.append(f"{dataset}/{video_id} ({path.name})")
                continue

            try:
                arr = np.load(path, allow_pickle=False)
                if arr.shape != EXPECTED_SHAPE:
                    invalid_files.append(
                        f"{path.name}: shape {arr.shape}, expected {EXPECTED_SHAPE}"
                    )
                    continue
                if arr.dtype != np.float32:
                    invalid_files.append(
                        f"{path.name}: dtype {arr.dtype}, expected float32"
                    )
                    continue
                if not np.isfinite(arr).all():
                    invalid_files.append(f"{path.name}: non-finite values")
                    continue
                if not np.any(arr):
                    invalid_files.append(f"{path.name}: all-zero sequence")
                    continue
            except Exception as exc:
                invalid_files.append(f"{path.name}: cannot load ({exc})")
                continue

            if key not in records_by_key:
                records_by_key[key] = {
                    "dataset": metadata.get("dataset", ""),
                    "split": metadata.get("split", ""),
                    "gloss": normalize(metadata.get("gloss")),
                    "label": metadata.get("label", ""),
                    "signer_id": metadata.get("signer_id", ""),
                    "start": metadata.get("start", ""),
                    "end": metadata.get("end", ""),
                    "video_id": video_id,
                    "keypoint_path": str(path.relative_to(PROJECT_ROOT)),
                }

    rows = sorted(
        records_by_key.values(),
        key=lambda r: (
            normalize(r["gloss"]),
            normalize(r["dataset"]),
            r["video_id"],
        ),
    )

    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    report = [
        "MASTER DATASET REVIEW",
        "=" * 60,
        f"Unique metadata dataset/video pairs: {len(metadata_by_key)}",
        f"Valid matched keypoint records: {len(rows)}",
        f"Duplicate/conflicting metadata entries: {len(duplicate_metadata)}",
        f"Files with invalid arrays or filenames: {len(invalid_files)}",
        f"Keypoints without matching metadata: {len(missing_metadata)}",
        "",
        "INVALID FILES (first 30)",
        *invalid_files[:30],
        "",
        "MISSING METADATA (first 30)",
        *missing_metadata[:30],
        "",
        "CONFLICTING METADATA (first 30)",
        *duplicate_metadata[:30],
        "",
        f"Review CSV: {OUTPUT_FILE}",
        "Original master_dataset.csv was not modified.",
    ]

    REPORT_FILE.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print(f"\nReport saved to: {REPORT_FILE}")


if __name__ == "__main__":
    main()
