import json
import re
from collections import defaultdict
from pathlib import Path


BASE = Path("data/metadata")

WLASL_FILE = BASE / "WLASL_v0.3.json"
MSASL_DIR = BASE / "msasl"

OUTPUT_FILE = BASE / "vocab_candidates_120.json"


def normalize(text):
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9 ]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


# ---------------------------------------------------------
# Load WLASL
# ---------------------------------------------------------

with open(WLASL_FILE, "r", encoding="utf-8") as f:
    wlasl = json.load(f)

wlasl_by_gloss = {
    normalize(item["gloss"]): item
    for item in wlasl
}


# ---------------------------------------------------------
# Load MS-ASL classes
# ---------------------------------------------------------

with open(MSASL_DIR / "MSASL_classes.json", "r", encoding="utf-8") as f:
    msasl_classes = json.load(f)

msasl_label_to_gloss = {
    i: normalize(gloss)
    for i, gloss in enumerate(msasl_classes)
}


# ---------------------------------------------------------
# Load MS-ASL records
# ---------------------------------------------------------

msasl_records = []

for split in ["train", "val", "test"]:

    with open(
        MSASL_DIR / f"MSASL_{split}.json",
        "r",
        encoding="utf-8"
    ) as f:
        records = json.load(f)

    for record in records:
        record["_split"] = split
        msasl_records.append(record)


# ---------------------------------------------------------
# Count MS-ASL clips and signers
# ---------------------------------------------------------

msasl_clips = defaultdict(int)
msasl_signers = defaultdict(set)

for record in msasl_records:

    label = record.get("label")

    if label not in msasl_label_to_gloss:
        continue

    gloss = msasl_label_to_gloss[label]

    msasl_clips[gloss] += 1

    signer_id = record.get("signer_id")

    if signer_id is not None:
        msasl_signers[gloss].add(
            f"MSASL_{signer_id}"
        )


# ---------------------------------------------------------
# Count WLASL clips and signers
# ---------------------------------------------------------

wlasl_clips = defaultdict(int)
wlasl_signers = defaultdict(set)

for gloss, item in wlasl_by_gloss.items():

    for instance in item.get("instances", []):

        wlasl_clips[gloss] += 1

        signer_id = instance.get("signer_id")

        if signer_id is not None:
            wlasl_signers[gloss].add(
                f"WLASL_{signer_id}"
            )


# ---------------------------------------------------------
# Find shared signs
# ---------------------------------------------------------

shared = (
    set(wlasl_by_gloss.keys())
    & set(msasl_label_to_gloss.values())
)


# ---------------------------------------------------------
# Build candidates
# ---------------------------------------------------------

candidates = []

for gloss in shared:

    w_count = wlasl_clips[gloss]
    m_count = msasl_clips[gloss]

    total = w_count + m_count

    signers = (
        wlasl_signers[gloss]
        | msasl_signers[gloss]
    )

    # Require reasonable representation from both datasets.
    # This is a candidate pool, not the final vocabulary.
    if w_count < 10 or m_count < 20:
        continue

    candidates.append({
        "gloss": gloss,
        "wlasl_clips": w_count,
        "msasl_clips": m_count,
        "total_metadata_clips": total,
        "wlasl_signers": len(wlasl_signers[gloss]),
        "msasl_signers": len(msasl_signers[gloss]),
        "total_signers": len(signers),
    })


# ---------------------------------------------------------
# Deterministic ranking
# ---------------------------------------------------------

candidates.sort(
    key=lambda x: (
        x["total_metadata_clips"],
        x["total_signers"],
        x["wlasl_clips"],
        x["msasl_clips"],
        x["gloss"],
    ),
    reverse=True,
)


# ---------------------------------------------------------
# Keep 120 candidates
# ---------------------------------------------------------

candidates = candidates[:120]


# ---------------------------------------------------------
# Add reproducible class IDs
# ---------------------------------------------------------

for class_id, item in enumerate(candidates):
    item["class_id"] = class_id


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        candidates,
        f,
        indent=2,
        ensure_ascii=False
    )


# ---------------------------------------------------------
# Report
# ---------------------------------------------------------

print("=" * 80)
print("VOCABULARY CANDIDATE SELECTION")
print("=" * 80)

print(f"Shared signs available: {len(shared)}")
print("Candidate requirements:")
print("  WLASL clips >= 10")
print("  MS-ASL clips >= 20")
print()
print(f"Candidates selected: {len(candidates)}")
print(f"Saved to: {OUTPUT_FILE}")
print()

print("First 20 candidates:")
print("-" * 80)

for item in candidates[:20]:

    print(
        f"{item['class_id']:3d}  "
        f"{item['gloss']:<20} "
        f"WL={item['wlasl_clips']:2d} "
        f"MS={item['msasl_clips']:2d} "
        f"Total={item['total_metadata_clips']:3d} "
        f"Signers={item['total_signers']:2d}"
    )