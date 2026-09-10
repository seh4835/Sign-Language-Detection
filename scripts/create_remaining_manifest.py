import csv
import json
from pathlib import Path


# =========================================================
# PROJECT ROOT
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# =========================================================
# INPUT FILES
# =========================================================

EXPANDED_MANIFEST = (
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

BATCH_02_LOG = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "balanced_batch_02_log.json"
)


# =========================================================
# OUTPUT
# =========================================================

OUTPUT_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "remaining_manifest.csv"
)


# =========================================================
# LOAD EXPANDED MANIFEST
# =========================================================

print("=" * 80)
print("CREATE REMAINING DOWNLOAD MANIFEST")
print("=" * 80)

print()
print("Loading expanded manifest...")

with open(
    EXPANDED_MANIFEST,
    "r",
    encoding="utf-8"
) as f:

    reader = csv.DictReader(f)
    all_records = list(reader)


print(
    f"Expanded manifest records: {len(all_records)}"
)


# =========================================================
# COLLECT ATTEMPTED VIDEO IDS
# =========================================================

attempted_video_ids = set()


# ---------------------------------------------------------
# Batch 01
# ---------------------------------------------------------

print()
print("Loading Batch 01 log...")

with open(
    BATCH_01_LOG,
    "r",
    encoding="utf-8"
) as f:

    batch_01 = json.load(f)


batch_01_ids = set()

for record in batch_01:

    video_id = str(
        record.get("video_id", "")
    ).strip()

    if video_id:
        batch_01_ids.add(video_id)
        attempted_video_ids.add(video_id)


print(
    f"Batch 01 attempted video IDs: "
    f"{len(batch_01_ids)}"
)


# ---------------------------------------------------------
# Batch 02
# ---------------------------------------------------------

print()
print("Loading Batch 02 log...")

with open(
    BATCH_02_LOG,
    "r",
    encoding="utf-8"
) as f:

    batch_02 = json.load(f)


batch_02_ids = set()

for record in batch_02:

    video_id = str(
        record.get("video_id", "")
    ).strip()

    if video_id:
        batch_02_ids.add(video_id)
        attempted_video_ids.add(video_id)


print(
    f"Batch 02 attempted video IDs: "
    f"{len(batch_02_ids)}"
)


# =========================================================
# OVERLAP CHECK
# =========================================================

overlap = (
    batch_01_ids
    & batch_02_ids
)

print()
print(
    f"Video IDs attempted in both batches: "
    f"{len(overlap)}"
)

if overlap:

    print(
        "WARNING: Some video IDs appear in both "
        "Batch 01 and Batch 02."
    )

else:

    print(
        "No overlap between Batch 01 and Batch 02."
    )


# =========================================================
# CREATE REMAINING RECORDS
# =========================================================

remaining = []

already_seen_manifest_ids = set()


for record in all_records:

    video_id = str(
        record.get("video_id", "")
    ).strip()

    if not video_id:
        continue


    # -----------------------------------------------------
    # Skip anything already attempted
    # -----------------------------------------------------

    if video_id in attempted_video_ids:
        continue


    # -----------------------------------------------------
    # Avoid duplicate records inside remaining manifest
    # -----------------------------------------------------

    if video_id in already_seen_manifest_ids:
        continue

    already_seen_manifest_ids.add(
        video_id
    )

    remaining.append(record)


# =========================================================
# SAVE REMAINING MANIFEST
# =========================================================

OUTPUT_MANIFEST.parent.mkdir(
    parents=True,
    exist_ok=True
)


fieldnames = [
    "dataset",
    "split",
    "gloss",
    "label",
    "signer_id",
    "start",
    "end",
    "url",
    "video_id"
]


with open(
    OUTPUT_MANIFEST,
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
        remaining
    )


# =========================================================
# PER-DATASET COUNTS
# =========================================================

wlasl_remaining = sum(
    1
    for record in remaining
    if str(
        record.get("dataset", "")
    ).strip().lower()
    == "wlasl"
)

msasl_remaining = sum(
    1
    for record in remaining
    if str(
        record.get("dataset", "")
    ).strip().lower()
    == "msasl"
)


# =========================================================
# PER-SIGN COUNTS
# =========================================================

sign_counts = {}

for record in remaining:

    gloss = str(
        record.get("gloss", "")
    ).strip().lower()

    sign_counts[gloss] = (
        sign_counts.get(gloss, 0)
        + 1
    )


# =========================================================
# FINAL VALIDATION
# =========================================================

remaining_ids = {
    str(
        record.get("video_id", "")
    ).strip()
    for record in remaining
}


still_attempted = (
    remaining_ids
    & attempted_video_ids
)


# =========================================================
# REPORT
# =========================================================

print()
print("=" * 80)
print("REMAINING MANIFEST SUMMARY")
print("=" * 80)

print()
print(
    f"Original manifest records: "
    f"{len(all_records)}"
)

print(
    f"Batch 01 attempted: "
    f"{len(batch_01_ids)}"
)

print(
    f"Batch 02 attempted: "
    f"{len(batch_02_ids)}"
)

print(
    f"Unique attempted records: "
    f"{len(attempted_video_ids)}"
)

print(
    f"Remaining records: "
    f"{len(remaining)}"
)

print()

print(
    f"Remaining WLASL records: "
    f"{wlasl_remaining}"
)

print(
    f"Remaining MS-ASL records: "
    f"{msasl_remaining}"
)

print()

print(
    f"Remaining signs represented: "
    f"{len(sign_counts)}"
)

print(
    f"Duplicate IDs inside remaining: "
    f"{len(remaining) - len(remaining_ids)}"
)

print(
    f"Previously attempted IDs still present: "
    f"{len(still_attempted)}"
)


# =========================================================
# SIGN DISTRIBUTION
# =========================================================

print()
print("=" * 80)
print("REMAINING RECORDS PER SIGN")
print("=" * 80)

for gloss in sorted(sign_counts):

    print(
        f"{gloss:20s} "
        f"{sign_counts[gloss]:4d}"
    )


# =========================================================
# FINAL STATUS
# =========================================================

print()
print("=" * 80)

if (
    len(still_attempted) == 0
    and len(remaining) > 0
):

    print(
        "REMAINING MANIFEST VALIDATION: PASS"
    )

else:

    print(
        "REMAINING MANIFEST VALIDATION: REVIEW REQUIRED"
    )

print("=" * 80)

print()
print(
    f"Saved to:"
)

print(
    OUTPUT_MANIFEST
)

print()
print(
    "IMPORTANT: This script only creates the "
    "remaining manifest. No videos were downloaded."
)