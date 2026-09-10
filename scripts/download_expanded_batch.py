import csv
import json
import subprocess
import sys
from pathlib import Path


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "combined_expanded_manifest.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "expanded_batch_01"
)

LOG_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "expanded_batch_01_log.json"
)


# =========================================================
# SETTINGS
# =========================================================

BATCH_SIZE = 500


# =========================================================
# CREATE OUTPUT DIRECTORY
# =========================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# LOAD MANIFEST
# =========================================================

print("=" * 70)
print("EXPANDED DATASET DOWNLOAD - BATCH 01")
print("=" * 70)

print()
print("Loading expanded manifest...")

with open(
    MANIFEST_PATH,
    "r",
    encoding="utf-8"
) as f:
    reader = csv.DictReader(f)
    records = list(reader)

print(f"Total manifest records: {len(records)}")


# =========================================================
# SELECT BATCH
# =========================================================

batch = records[:BATCH_SIZE]

print(f"Records selected for this batch: {len(batch)}")
print(f"Output directory: {OUTPUT_DIR}")
print()


# =========================================================
# DOWNLOAD
# =========================================================

results = []

successful = 0
failed = 0

for index, record in enumerate(batch, start=1):

    dataset = record["dataset"].strip()
    gloss = record["gloss"].strip()
    split = record["split"].strip()
    video_id = record["video_id"].strip()
    url = record["url"].strip()

    filename = (
        f"{dataset}__"
        f"{gloss}__"
        f"{split}__"
        f"{video_id}.mp4"
    )

    output_path = OUTPUT_DIR / filename

    print(
        f"[{index:03d}/{len(batch)}] "
        f"{dataset} | {gloss} | {video_id}"
    )

    # -----------------------------------------------------
    # Skip if already downloaded
    # -----------------------------------------------------

    if output_path.exists() and output_path.stat().st_size > 0:

        print("       EXISTS - skipped")

        results.append({
            "dataset": dataset,
            "gloss": gloss,
            "split": split,
            "signer_id": record["signer_id"],
            "video_id": video_id,
            "url": url,
            "filename": filename,
            "status": "exists"
        })

        successful += 1
        continue

    # -----------------------------------------------------
    # yt-dlp command
    # -----------------------------------------------------

    command = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--no-playlist",
        "--quiet",
        "--no-warnings",
        "-f",
        "mp4/best",
        "-o",
        str(output_path),
        url
    ]

    try:

        process = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if (
            process.returncode == 0
            and output_path.exists()
            and output_path.stat().st_size > 0
        ):

            print("       SUCCESS")

            status = "success"
            successful += 1

        else:

            print("       FAILED")

            status = "failed"
            failed += 1

            # Remove incomplete file if created
            if output_path.exists():
                try:
                    output_path.unlink()
                except Exception:
                    pass

    except Exception as e:

        print(f"       ERROR: {e}")

        status = "error"
        failed += 1

    results.append({
        "dataset": dataset,
        "gloss": gloss,
        "split": split,
        "signer_id": record["signer_id"],
        "video_id": video_id,
        "url": url,
        "filename": filename,
        "status": status
    })


# =========================================================
# SAVE LOG
# =========================================================

with open(
    LOG_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=2
    )


# =========================================================
# FINAL SUMMARY
# =========================================================

print()
print("=" * 70)
print("BATCH 01 DOWNLOAD SUMMARY")
print("=" * 70)

print(f"Attempted:  {len(batch)}")
print(f"Successful: {successful}")
print(f"Failed:     {failed}")

print()
print(f"Videos saved to:")
print(OUTPUT_DIR)

print()
print(f"Log saved to:")
print(LOG_PATH)

print()
print("=" * 70)
print("BATCH 01 COMPLETE")
print("=" * 70)