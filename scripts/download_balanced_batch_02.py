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
    / "balanced_batch_02_manifest.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "balanced_batch_02"
)

LOG_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "balanced_batch_02_log.json"
)


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
print("BALANCED BATCH 02 DOWNLOAD")
print("=" * 70)

print()
print("Loading Batch 02 manifest...")

with open(
    MANIFEST_PATH,
    "r",
    encoding="utf-8"
) as f:

    reader = csv.DictReader(f)
    records = list(reader)

print(
    f"Records to attempt: {len(records)}"
)

print(
    f"Output directory: {OUTPUT_DIR}"
)

print()


# =========================================================
# DOWNLOAD
# =========================================================

results = []

successful = 0
failed = 0


for index, record in enumerate(
    records,
    start=1
):

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

    output_path = (
        OUTPUT_DIR
        / filename
    )

    print(
        f"[{index:04d}/{len(records)}] "
        f"{dataset} | {gloss} | {video_id}"
    )


    # -----------------------------------------------------
    # Skip existing valid-looking files
    # -----------------------------------------------------

    if (
        output_path.exists()
        and output_path.stat().st_size > 0
    ):

        print(
            "       EXISTS - skipped"
        )

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


    # -----------------------------------------------------
    # Run downloader
    # -----------------------------------------------------

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

            print(
                "       SUCCESS"
            )

            status = "success"

            successful += 1

        else:

            print(
                "       FAILED"
            )

            status = "failed"

            failed += 1


            # Remove incomplete file
            if output_path.exists():

                try:
                    output_path.unlink()

                except Exception:
                    pass


    except Exception as e:

        print(
            f"       ERROR: {e}"
        )

        status = "error"

        failed += 1


    # -----------------------------------------------------
    # Save result
    # -----------------------------------------------------

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
print("BATCH 02 DOWNLOAD SUMMARY")
print("=" * 70)

print(
    f"Attempted:  {len(records)}"
)

print(
    f"Successful: {successful}"
)

print(
    f"Failed:     {failed}"
)

if len(records) > 0:

    success_rate = (
        successful
        / len(records)
        * 100
    )

    print(
        f"Success rate: {success_rate:.2f}%"
    )

print()

print(
    "Videos saved to:"
)

print(
    OUTPUT_DIR
)

print()

print(
    "Log saved to:"
)

print(
    LOG_PATH
)

print()
print("=" * 70)
print("BATCH 02 DOWNLOAD COMPLETE")
print("=" * 70)