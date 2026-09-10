import json
import csv
from pathlib import Path
from urllib.parse import parse_qs, urlparse


BASE = Path("data")
METADATA = BASE / "metadata"
MSASL_DIR = METADATA / "msasl"

VOCAB_FILE = METADATA / "vocab_candidates_120.json"
OUTPUT_FILE = METADATA / "candidate_download_manifest.csv"

MSASL_SPLITS = ["train", "val", "test"]

# Initial screening target
TARGET_PER_SIGN = 20


def get_video_id(url):
    """Extract YouTube video ID."""
    try:
        query = parse_qs(urlparse(url).query)
        return query.get("v", [None])[0]
    except Exception:
        return None


# ---------------------------------------------------------
# Load 120 candidate signs
# ---------------------------------------------------------

with open(VOCAB_FILE, "r", encoding="utf-8") as f:
    candidates = json.load(f)

candidate_glosses = {
    item["gloss"].lower().strip()
    for item in candidates
}


# ---------------------------------------------------------
# Load MS-ASL classes
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Collect MS-ASL records
# ---------------------------------------------------------

records = []

for split in MSASL_SPLITS:

    path = MSASL_DIR / f"MSASL_{split}.json"

    with open(path, "r", encoding="utf-8") as f:
        split_records = json.load(f)

    for record in split_records:

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

        records.append({
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


# ---------------------------------------------------------
# Remove duplicate videos
# ---------------------------------------------------------

unique_records = []
seen = set()

for record in records:

    video_id = record["video_id"]

    if video_id in seen:
        continue

    seen.add(video_id)
    unique_records.append(record)


# ---------------------------------------------------------
# Group by gloss
# ---------------------------------------------------------

by_gloss = {}

for record in unique_records:

    gloss = record["gloss"]

    by_gloss.setdefault(gloss, []).append(record)


# ---------------------------------------------------------
# Deterministic ordering
# ---------------------------------------------------------

for gloss in by_gloss:

    by_gloss[gloss].sort(
        key=lambda x: (
            x["split"],
            str(x["signer_id"]),
            x["video_id"]
        )
    )


# ---------------------------------------------------------
# Select initial 20 per sign
# ---------------------------------------------------------

manifest = []

for gloss in sorted(candidate_glosses):

    available = by_gloss.get(gloss, [])

    selected = available[:TARGET_PER_SIGN]

    for record in selected:
        manifest.append(record)


# ---------------------------------------------------------
# Save manifest
# ---------------------------------------------------------

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
    writer.writerows(manifest)


# ---------------------------------------------------------
# Report
# ---------------------------------------------------------

print("=" * 80)
print("CANDIDATE DOWNLOAD MANIFEST")
print("=" * 80)

print(f"Candidate signs: {len(candidate_glosses)}")
print(f"Target clips/sign: {TARGET_PER_SIGN}")
print(f"Manifest records: {len(manifest)}")

print()

print("Per-sign availability:")

insufficient = []

for gloss in sorted(candidate_glosses):

    count = len(by_gloss.get(gloss, []))

    status = "OK" if count >= TARGET_PER_SIGN else "LOW"

    print(
        f"{gloss:20s} "
        f"available={count:3d} "
        f"selected={min(count, TARGET_PER_SIGN):2d} "
        f"{status}"
    )

    if count < TARGET_PER_SIGN:
        insufficient.append((gloss, count))


print()
print("=" * 80)
print("MANIFEST SUMMARY")
print("=" * 80)

print(f"Signs with >= {TARGET_PER_SIGN} available: "
      f"{len(candidate_glosses) - len(insufficient)}")

print(f"Signs with < {TARGET_PER_SIGN} available: "
      f"{len(insufficient)}")

print(f"Saved to: {OUTPUT_FILE}")

if insufficient:

    print()
    print("SIGNS NEEDING EXTRA SOURCES:")
    for gloss, count in insufficient:
        print(f"  {gloss}: {count}")