import csv
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse


# =========================================================
# PATHS
# =========================================================

BASE = Path("data")
METADATA = BASE / "metadata"

WLASL_FILE = METADATA / "WLASL_v0.3.json"
MSASL_DIR = METADATA / "msasl"
VOCAB_FILE = METADATA / "vocab_candidates_120.json"

# IMPORTANT:
# This is a NEW file.
# The original 2,400-row manifest will NOT be changed.
OUTPUT_FILE = METADATA / "combined_expanded_manifest.csv"


# =========================================================
# SETTINGS
# =========================================================

# We want up to 40 acquisition candidates per sign.
TARGET_PER_SIGN = 40

MSASL_SPLITS = [
    "train",
    "val",
    "test"
]


# =========================================================
# VIDEO ID EXTRACTION
# =========================================================

def get_video_id(url):
    """Extract YouTube video ID from a URL."""

    try:
        query = parse_qs(
            urlparse(url).query
        )

        return query.get(
            "v",
            [None]
        )[0]

    except Exception:
        return None


# =========================================================
# LOAD CANDIDATE VOCABULARY
# =========================================================

print("=" * 80)
print("EXPANDED COMBINED WLASL + MS-ASL MANIFEST")
print("=" * 80)

print()
print("Loading candidate vocabulary...")

with open(
    VOCAB_FILE,
    "r",
    encoding="utf-8"
) as f:
    candidates = json.load(f)


candidate_glosses = {
    item["gloss"].lower().strip()
    for item in candidates
}


print(
    f"Candidate signs: {len(candidate_glosses)}"
)


# =========================================================
# LOAD WLASL
# =========================================================

print()
print("Loading WLASL metadata...")

with open(
    WLASL_FILE,
    "r",
    encoding="utf-8"
) as f:
    wlasl = json.load(f)


print(
    f"WLASL signs: {len(wlasl)}"
)


# =========================================================
# COLLECT WLASL RECORDS
# =========================================================

wlasl_records = []

for item in wlasl:

    gloss = str(
        item.get("gloss", "")
    ).lower().strip()

    if gloss not in candidate_glosses:
        continue

    instances = item.get(
        "instances",
        []
    )

    for instance in instances:

        url = instance.get("url")

        if not url:
            continue

        video_id = instance.get(
            "video_id"
        )

        if not video_id:
            video_id = get_video_id(url)

        if not video_id:
            continue

        wlasl_records.append({
            "dataset": "wlasl",
            "split": instance.get("split"),
            "gloss": gloss,
            "label": None,
            "signer_id": instance.get("signer_id"),
            "start": None,
            "end": None,
            "url": url,
            "video_id": str(video_id)
        })


print(
    f"WLASL matching records: "
    f"{len(wlasl_records)}"
)


# =========================================================
# LOAD MS-ASL CLASSES
# =========================================================

print()
print("Loading MS-ASL classes...")

with open(
    MSASL_DIR / "MSASL_classes.json",
    "r",
    encoding="utf-8"
) as f:
    classes = json.load(f)


label_to_gloss = {
    i: str(gloss).lower().strip()
    for i, gloss in enumerate(classes)
}


# =========================================================
# COLLECT MS-ASL RECORDS
# =========================================================

msasl_records = []

for split in MSASL_SPLITS:

    path = (
        MSASL_DIR
        / f"MSASL_{split}.json"
    )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        records = json.load(f)

    for record in records:

        label = record.get("label")

        if label not in label_to_gloss:
            continue

        gloss = label_to_gloss[label]

        if gloss not in candidate_glosses:
            continue

        url = record.get("url")

        if not url:
            continue

        video_id = get_video_id(url)

        if not video_id:
            continue

        msasl_records.append({
            "dataset": "msasl",
            "split": split,
            "gloss": gloss,
            "label": label,
            "signer_id": record.get("signer_id"),
            "start": record.get("start"),
            "end": record.get("end"),
            "url": url,
            "video_id": video_id
        })


print(
    f"MS-ASL matching records: "
    f"{len(msasl_records)}"
)


# =========================================================
# COMBINE
# =========================================================

all_records = (
    wlasl_records
    + msasl_records
)


print()
print(
    f"Combined records before "
    f"duplicate removal: {len(all_records)}"
)


# =========================================================
# REMOVE DUPLICATE VIDEO IDs
# =========================================================

unique_records = []

seen_video_ids = set()

for record in all_records:

    video_id = record["video_id"]

    if video_id in seen_video_ids:
        continue

    seen_video_ids.add(video_id)

    unique_records.append(record)


print(
    f"Combined records after "
    f"duplicate removal: "
    f"{len(unique_records)}"
)


# =========================================================
# GROUP BY GLOSS
# =========================================================

by_gloss = {}

for record in unique_records:

    gloss = record["gloss"]

    by_gloss.setdefault(
        gloss,
        []
    ).append(record)


# =========================================================
# SELECT RECORDS WITH DIVERSITY
#
# Instead of simply taking the first 40 records,
# we select across different signer/dataset groups.
# =========================================================

def select_diverse_records(records, target):

    # -----------------------------------------------------
    # Create signer/dataset groups
    # -----------------------------------------------------

    groups = {}

    for record in records:

        signer = str(
            record["signer_id"]
        )

        dataset = str(
            record["dataset"]
        )

        group_key = (
            dataset,
            signer
        )

        groups.setdefault(
            group_key,
            []
        ).append(record)


    # -----------------------------------------------------
    # Sort records within each group
    # -----------------------------------------------------

    for group_key in groups:

        groups[group_key].sort(
            key=lambda x: (
                str(x["split"]),
                str(x["video_id"])
            )
        )


    # -----------------------------------------------------
    # Round-robin selection
    #
    # This spreads selected records across
    # different signer/dataset groups.
    # -----------------------------------------------------

    selected = []

    group_keys = sorted(
        groups.keys()
    )

    position = 0

    while (
        len(selected) < target
        and group_keys
    ):

        added_this_round = False

        for group_key in group_keys:

            group_records = groups[group_key]

            if position < len(group_records):

                selected.append(
                    group_records[position]
                )

                added_this_round = True

                if len(selected) >= target:
                    break

        if not added_this_round:
            break

        position += 1


    return selected


# =========================================================
# CREATE EXPANDED MANIFEST
# =========================================================

manifest = []

print()
print("=" * 80)
print("SELECTING EXPANDED ACQUISITION POOL")
print("=" * 80)
print()

for gloss in sorted(candidate_glosses):

    available = by_gloss.get(
        gloss,
        []
    )

    selected = select_diverse_records(
        available,
        TARGET_PER_SIGN
    )

    manifest.extend(
        selected
    )

    print(
        f"{gloss:20s} "
        f"available={len(available):3d} "
        f"selected={len(selected):3d}"
    )


# =========================================================
# SAVE CSV
# =========================================================

OUTPUT_FILE.parent.mkdir(
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
        manifest
    )


# =========================================================
# FINAL REPORT
# =========================================================

print()
print("=" * 80)
print("EXPANDED MANIFEST SUMMARY")
print("=" * 80)
print()

ready_40 = 0
between_15_39 = 0
below_15 = 0

for gloss in sorted(candidate_glosses):

    available = by_gloss.get(
        gloss,
        []
    )

    count = len(available)

    if count >= 40:

        status = "40+"
        ready_40 += 1

    elif count >= 15:

        status = "15-39"
        between_15_39 += 1

    else:

        status = "<15"
        below_15 += 1

    print(
        f"{gloss:20s} "
        f"available={count:3d} "
        f"status={status}"
    )


print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)

print(
    f"Candidate signs: "
    f"{len(candidate_glosses)}"
)

print(
    f"Signs with >=40 available records: "
    f"{ready_40}"
)

print(
    f"Signs with 15-39 available records: "
    f"{between_15_39}"
)

print(
    f"Signs with <15 available records: "
    f"{below_15}"
)

print(
    f"Expanded manifest records: "
    f"{len(manifest)}"
)

print(
    f"Saved to: "
    f"{OUTPUT_FILE}"
)

print()

print(
    "IMPORTANT: These are metadata-available "
    "records, NOT confirmed usable videos."
)

print(
    "Video download, cleaning, keypoint quality "
    "and signer-disjoint evaluation will determine "
    "the final usable dataset."
)

print()

print(
    "Original combined_download_manifest.csv "
    "was NOT modified."
)