import csv
from pathlib import Path
from collections import defaultdict


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

BATCH_01_LOG = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "expanded_batch_01_log.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "balanced_batch_02_manifest.csv"
)


# =========================================================
# SETTINGS
# =========================================================

RECORDS_PER_SIGN = 15


# =========================================================
# LOAD EXPANDED MANIFEST
# =========================================================

print("=" * 80)
print("CREATE BALANCED BATCH 02")
print("=" * 80)

print()
print("Loading expanded manifest...")

with open(
    MANIFEST_PATH,
    "r",
    encoding="utf-8"
) as f:

    reader = csv.DictReader(f)
    manifest = list(reader)

print(
    f"Expanded manifest records: {len(manifest)}"
)


# =========================================================
# LOAD BATCH 01 LOG
# =========================================================

print()
print("Loading Batch 01 log...")

import json

with open(
    BATCH_01_LOG,
    "r",
    encoding="utf-8"
) as f:

    batch_01_results = json.load(f)

print(
    f"Batch 01 records: {len(batch_01_results)}"
)


# =========================================================
# BUILD SET OF ALREADY-ATTEMPTED RECORDS
# =========================================================

already_attempted = set()

for result in batch_01_results:

    filename = str(
        result.get("filename", "")
    ).strip()

    if filename:
        already_attempted.add(filename)


print(
    f"Already attempted in Batch 01: "
    f"{len(already_attempted)}"
)


# =========================================================
# BUILD EXPECTED FILENAME
# =========================================================

def make_filename(row):

    dataset = str(
        row["dataset"]
    ).strip().lower()

    gloss = str(
        row["gloss"]
    ).strip()

    split = str(
        row["split"]
    ).strip().lower()

    video_id = str(
        row["video_id"]
    ).strip()

    if video_id.endswith(".0"):
        video_id = video_id[:-2]

    return (
        f"{dataset}__"
        f"{gloss}__"
        f"{split}__"
        f"{video_id}.mp4"
    )


# =========================================================
# REMOVE BATCH 01 RECORDS
# =========================================================

remaining_records = []

for row in manifest:

    filename = make_filename(row)

    if filename not in already_attempted:

        row = dict(row)
        row["filename"] = filename

        remaining_records.append(row)


print(
    f"Records remaining after Batch 01: "
    f"{len(remaining_records)}"
)


# =========================================================
# GROUP REMAINING RECORDS BY SIGN
# =========================================================

by_gloss = defaultdict(list)

for row in remaining_records:

    gloss = row["gloss"].strip()

    by_gloss[gloss].append(row)


# =========================================================
# SORT WITH SIGNER/DATASET DIVERSITY
# =========================================================

for gloss in by_gloss:

    by_gloss[gloss].sort(
        key=lambda x: (
            str(x["signer_id"]),
            str(x["dataset"]),
            str(x["split"]),
            str(x["video_id"])
        )
    )


# =========================================================
# CREATE BALANCED BATCH
# =========================================================

all_signs = sorted(
    by_gloss.keys()
)

batch_02 = []

print()
print("=" * 80)
print("BATCH 02 SELECTION")
print("=" * 80)
print()

for gloss in all_signs:

    available = by_gloss[gloss]

    selected = available[
        :RECORDS_PER_SIGN
    ]

    batch_02.extend(selected)

    print(
        f"{gloss:20s} "
        f"available={len(available):3d} "
        f"selected={len(selected):2d}"
    )


# =========================================================
# SAVE BATCH 02
# =========================================================

fieldnames = [
    "dataset",
    "split",
    "gloss",
    "label",
    "signer_id",
    "start",
    "end",
    "url",
    "video_id",
    "filename"
]

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    OUTPUT_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(
        batch_02
    )


# =========================================================
# FINAL SUMMARY
# =========================================================

print()
print("=" * 80)
print("BATCH 02 SUMMARY")
print("=" * 80)

print(
    f"Signs included: "
    f"{len(all_signs)}"
)

print(
    f"Target records per sign: "
    f"{RECORDS_PER_SIGN}"
)

print(
    f"Batch 02 records: "
    f"{len(batch_02)}"
)

print()
print(
    f"Saved to:"
)

print(
    OUTPUT_PATH
)

print()

print(
    "Batch 01 records were excluded."
)

print(
    "Batch 02 is balanced across the remaining signs."
)

print()
print("=" * 80)
print("BALANCED BATCH 02 CREATED")
print("=" * 80)