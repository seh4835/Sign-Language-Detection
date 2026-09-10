import csv
import json
from pathlib import Path


# =========================================================
# PROJECT ROOT
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# =========================================================
# PATHS
# =========================================================

METADATA_DIR = PROJECT_ROOT / "data" / "metadata"

VOCAB_FILE = (
    METADATA_DIR / "vocab_candidates_120.json"
)

EXPANDED_MANIFEST = (
    METADATA_DIR / "combined_expanded_manifest.csv"
)

MASTER_INVENTORY = (
    METADATA_DIR / "master_dataset_inventory.csv"
)

OUTPUT_FILE = (
    METADATA_DIR / "recovery_manifest.csv"
)


# =========================================================
# LOAD CANDIDATE VOCABULARY
# =========================================================

print("=" * 80)
print("RECOVERY MANIFEST CREATION")
print("=" * 80)

print()
print("Loading candidate vocabulary...")

with open(
    VOCAB_FILE,
    "r",
    encoding="utf-8"
) as f:

    candidates = json.load(f)


candidate_signs = {
    str(item["gloss"]).strip().lower()
    for item in candidates
}


print(
    f"Candidate signs: {len(candidate_signs)}"
)


# =========================================================
# LOAD EXPANDED MANIFEST
# =========================================================

print()
print("Loading expanded manifest...")

with open(
    EXPANDED_MANIFEST,
    "r",
    encoding="utf-8"
) as f:

    reader = csv.DictReader(f)
    manifest_records = list(reader)


print(
    f"Expanded manifest records: "
    f"{len(manifest_records)}"
)


# =========================================================
# LOAD ALREADY DOWNLOADED RECORDS
# =========================================================

print()
print("Loading master inventory...")

with open(
    MASTER_INVENTORY,
    "r",
    encoding="utf-8"
) as f:

    reader = csv.DictReader(f)
    downloaded_records = list(reader)


print(
    f"Downloaded inventory records: "
    f"{len(downloaded_records)}"
)


# =========================================================
# BUILD DOWNLOADED VIDEO ID SET
# =========================================================

downloaded_ids = set()

for record in downloaded_records:

    video_id = str(
        record.get(
            "video_id",
            ""
        )
    ).strip()

    if video_id:
        downloaded_ids.add(video_id)


print(
    f"Already downloaded video IDs: "
    f"{len(downloaded_ids)}"
)


# =========================================================
# CREATE RECOVERY POOL
# =========================================================

recovery_records = []

seen_ids = set()


for record in manifest_records:

    gloss = (
        str(
            record.get(
                "gloss",
                ""
            )
        )
        .strip()
        .lower()
    )

    if gloss not in candidate_signs:
        continue


    video_id = str(
        record.get(
            "video_id",
            ""
        )
    ).strip()

    if not video_id:
        continue


    # -----------------------------------------------------
    # Skip videos already downloaded
    # -----------------------------------------------------

    if video_id in downloaded_ids:
        continue


    # -----------------------------------------------------
    # Skip duplicate IDs
    # -----------------------------------------------------

    if video_id in seen_ids:
        continue

    seen_ids.add(video_id)


    recovery_records.append(record)


# =========================================================
# SORT FOR CONTROLLED ACQUISITION
# =========================================================

recovery_records.sort(
    key=lambda x: (
        str(
            x.get(
                "gloss",
                ""
            )
        ).lower(),

        str(
            x.get(
                "dataset",
                ""
            )
        ).lower(),

        str(
            x.get(
                "split",
                ""
            )
        ).lower(),

        str(
            x.get(
                "signer_id",
                ""
            )
        ),

        str(
            x.get(
                "video_id",
                ""
            )
        )
    )
)


# =========================================================
# SAVE MANIFEST
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
    "video_id"
]


OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


with open(
    OUTPUT_FILE,
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
        recovery_records
    )


# =========================================================
# PER-SIGN SUMMARY
# =========================================================

from collections import Counter

sign_counts = Counter(
    record["gloss"]
    for record in recovery_records
)


print()
print("=" * 80)
print("RECOVERY POOL BY SIGN")
print("=" * 80)

for gloss in sorted(candidate_signs):

    count = sign_counts.get(
        gloss,
        0
    )

    print(
        f"{gloss:20s} {count:4d}"
    )


# =========================================================
# DATASET SUMMARY
# =========================================================

dataset_counts = Counter(
    record["dataset"]
    for record in recovery_records
)


print()
print("=" * 80)
print("RECOVERY MANIFEST SUMMARY")
print("=" * 80)

print()

print(
    f"Candidate signs: "
    f"{len(candidate_signs)}"
)

print(
    f"Expanded manifest records: "
    f"{len(manifest_records)}"
)

print(
    f"Already downloaded records: "
    f"{len(downloaded_ids)}"
)

print(
    f"Recovery records: "
    f"{len(recovery_records)}"
)

print()

print(
    f"MS-ASL recovery records: "
    f"{dataset_counts.get('msasl', 0)}"
)

print(
    f"WLASL recovery records: "
    f"{dataset_counts.get('wlasl', 0)}"
)


# =========================================================
# SIGNS WITH NO RECOVERY RECORDS
# =========================================================

no_recovery = [
    gloss
    for gloss in sorted(candidate_signs)
    if sign_counts.get(gloss, 0) == 0
]


print()

print(
    f"Signs with no recovery records: "
    f"{len(no_recovery)}"
)

if no_recovery:

    for gloss in no_recovery:
        print(
            f"  - {gloss}"
        )


# =========================================================
# FINAL
# =========================================================

print()
print("=" * 80)
print("RECOVERY MANIFEST VALIDATION")
print("=" * 80)

expected_minimum = len(
    set(
        record["video_id"]
        for record in recovery_records
    )
)

actual_records = len(
    recovery_records
)


if actual_records == expected_minimum:

    print()
    print(
        "No duplicate video IDs found."
    )

    print(
        "RECOVERY MANIFEST VALIDATION: PASS"
    )

else:

    print()
    print(
        "RECOVERY MANIFEST VALIDATION: FAIL"
    )


print()
print("Saved to:")
print(OUTPUT_FILE)

print()
print(
    "IMPORTANT: No videos were downloaded."
)

print(
    "This manifest contains only records "
    "not already downloaded."
)