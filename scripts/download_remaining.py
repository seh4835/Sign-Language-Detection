import csv
import json
import subprocess
import sys
from pathlib import Path


# =========================================================
# PROJECT ROOT
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# =========================================================
# PATHS
# =========================================================

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "remaining_manifest.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "remaining_downloads"
)

LOG_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "remaining_downloads_log.json"
)


# =========================================================
# SETTINGS
# =========================================================

BATCH_SIZE = 500


# =========================================================
# PREPARE OUTPUT
# =========================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# LOAD MANIFEST
# =========================================================

print("=" * 80)
print("REMAINING DATASET DOWNLOAD")
print("=" * 80)

print()
print("Loading remaining manifest...")

with open(
    MANIFEST_PATH,
    "r",
    encoding="utf-8"
) as f:

    reader = csv.DictReader(f)
    records = list(reader)


print(
    f"Remaining records: {len(records)}"
)

print(
    f"Internal batch size: {BATCH_SIZE}"
)

print(
    f"Output directory: {OUTPUT_DIR}"
)

print()


# =========================================================
# LOAD EXISTING LOG IF PRESENT
# =========================================================

if LOG_PATH.exists():

    with open(
        LOG_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        results = json.load(f)

else:

    results = []


# =========================================================
# EXISTING VIDEO IDS
# =========================================================

existing_results = {
    str(
        item.get("video_id", "")
    ).strip()
    for item in results
    if item.get("video_id")
}


# =========================================================
# DOWNLOAD
# =========================================================

successful = 0
failed = 0
skipped = 0


for index, record in enumerate(
    records,
    start=1
):

    dataset = str(
        record.get("dataset", "")
    ).strip()

    split = str(
        record.get("split", "")
    ).strip()

    gloss = str(
        record.get("gloss", "")
    ).strip()

    signer_id = str(
        record.get("signer_id", "")
    ).strip()

    video_id = str(
        record.get("video_id", "")
    ).strip()

    url = str(
        record.get("url", "")
    ).strip()


    # -----------------------------------------------------
    # Filename
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # Progress
    # -----------------------------------------------------

    print(
        f"[{index:04d}/{len(records)}] "
        f"{dataset} | {gloss} | {video_id}"
    )


    # -----------------------------------------------------
    # Skip if already successfully processed
    # -----------------------------------------------------

    if (
        video_id in existing_results
        and output_path.exists()
        and output_path.stat().st_size > 0
    ):

        print(
            "       SKIPPED - already processed"
        )

        skipped += 1
        continue


    # -----------------------------------------------------
    # Skip if file already exists
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
            "split": split,
            "gloss": gloss,
            "signer_id": signer_id,
            "video_id": video_id,
            "url": url,
            "filename": filename,
            "status": "exists"
        })

        existing_results.add(video_id)

        skipped += 1
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
    # Download
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

            existing_results.add(
                video_id
            )

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
        "split": split,
        "gloss": gloss,
        "signer_id": signer_id,
        "video_id": video_id,
        "url": url,
        "filename": filename,
        "status": status
    })


    # -----------------------------------------------------
    # Periodic log save
    # -----------------------------------------------------

    if index % 50 == 0:

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

        print(
            "       Progress log saved."
        )


# =========================================================
# FINAL LOG SAVE
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
print("=" * 80)
print("REMAINING DOWNLOAD SUMMARY")
print("=" * 80)

print(
    f"Manifest records: {len(records)}"
)

print(
    f"Successful:       {successful}"
)

print(
    f"Failed:           {failed}"
)

print(
    f"Skipped:          {skipped}"
)

total_successful = (
    successful + skipped
)

print(
    f"Total available after this run: "
    f"{total_successful}"
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
print("=" * 80)
print("REMAINING DOWNLOAD COMPLETE")
print("=" * 80)