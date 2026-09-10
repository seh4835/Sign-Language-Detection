from pathlib import Path
import json
import hashlib

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MASTER_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "master_dataset.csv"
)

VOCAB_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "vocab_candidates_120.json"
)

KEYPOINT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "keypoints"
)

OUTPUT_DATASET = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "final_cleaned_dataset.csv"
)

OUTPUT_SUMMARY = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "final_cleaning_summary.csv"
)

OUTPUT_REPORT = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "final_cleaning_report.json"
)


# ============================================================
# SETTINGS
# ============================================================

EXPECTED_SHAPE = (30, 1662)
MIN_CLIPS_PER_SIGN = 15


# ============================================================
# FIND KEYPOINT FILE
# ============================================================

def find_keypoint_file(row):
    """
    Find the keypoint file for one master-dataset record.

    First, use the keypoint_path stored in the master dataset
    if available.

    If that is unavailable, search the complete data/keypoints
    directory using the video_id.
    """

    # --------------------------------------------------------
    # METHOD 1:
    # Use keypoint_path already stored in master dataset
    # --------------------------------------------------------

    if "keypoint_path" in row.index:

        stored_path = str(row["keypoint_path"]).strip()

        if stored_path and stored_path.lower() != "nan":

            candidate = PROJECT_ROOT / stored_path

            if candidate.exists():

                return candidate


    # --------------------------------------------------------
    # METHOD 2:
    # Search the complete keypoints directory
    # --------------------------------------------------------

    video_id = str(row["video_id"]).strip()

    if not KEYPOINT_ROOT.exists():

        return None

    matches = []

    for path in KEYPOINT_ROOT.rglob("*.npy"):

        parts = path.stem.split("__")

        # Expected format:
        #
        # dataset__gloss__split__video_id
        #
        # Example:
        #
        # wlasl__again__test__01457.npy

        if len(parts) >= 4:

            file_video_id = parts[3]

            if file_video_id == video_id:

                matches.append(path)

    if not matches:

        return None

    # Deterministic choice if more than one exists.
    matches = sorted(matches)

    return matches[0]


# ============================================================
# KEYPOINT HASH
# ============================================================

def keypoint_hash(array):
    """
    Generate a SHA-256 hash of the complete keypoint array.

    Exact identical sequences are treated as duplicate samples.
    """

    return hashlib.sha256(
        array.tobytes()
    ).hexdigest()


# ============================================================
# START
# ============================================================

print("=" * 75)
print("FINAL DATASET CLEANING")
print("=" * 75)


# ============================================================
# LOAD MASTER DATASET
# ============================================================

print()
print("Loading master dataset...")

if not MASTER_PATH.exists():

    print()
    print("ERROR: master_dataset.csv was not found.")
    print(MASTER_PATH)
    raise SystemExit

master = pd.read_csv(MASTER_PATH)

print("Master records:", len(master))


# ============================================================
# CHECK MASTER COLUMNS
# ============================================================

required_columns = [
    "video_id",
    "gloss",
    "dataset",
    "split",
    "signer_id",
]

missing_columns = [
    column
    for column in required_columns
    if column not in master.columns
]

if missing_columns:

    print()
    print("ERROR: Required columns are missing:")
    print(missing_columns)
    print()
    print("Available columns:")
    print(list(master.columns))
    raise SystemExit


# ============================================================
# LOAD CANDIDATE VOCABULARY
# ============================================================

print()
print("Loading candidate vocabulary...")

if not VOCAB_PATH.exists():

    print()
    print("ERROR: vocab_candidates_120.json was not found.")
    print(VOCAB_PATH)
    raise SystemExit

with open(
    VOCAB_PATH,
    "r",
    encoding="utf-8"
) as file:

    vocab_data = json.load(file)


# ============================================================
# READ VOCABULARY
# ============================================================

candidate_glosses = []

if isinstance(vocab_data, list):

    for item in vocab_data:

        if isinstance(item, str):

            candidate_glosses.append(
                item.lower().strip()
            )

        elif isinstance(item, dict):

            if "gloss" in item:

                candidate_glosses.append(
                    str(item["gloss"])
                    .lower()
                    .strip()
                )

candidate_glosses = sorted(
    set(candidate_glosses)
)

print(
    "Candidate signs:",
    len(candidate_glosses)
)


# ============================================================
# NORMALIZE GLOSS
# ============================================================

master["gloss"] = (
    master["gloss"]
    .astype(str)
    .str.lower()
    .str.strip()
)


# ============================================================
# SELECT CANDIDATE RECORDS
# ============================================================

candidate_set = set(candidate_glosses)

candidate_records = master[
    master["gloss"].isin(candidate_set)
].copy()

print(
    "Records matching candidate vocabulary:",
    len(candidate_records)
)


# ============================================================
# CLEANING
# ============================================================

clean_records = []

rejection_counts = {
    "missing_keypoints": 0,
    "wrong_shape": 0,
    "non_finite": 0,
    "all_zero": 0,
    "duplicate_sequence": 0,
}

seen_hashes = set()


print()
print("Checking keypoint sequences...")
print()


# ============================================================
# PROCESS EACH RECORD
# ============================================================

for _, row in candidate_records.iterrows():

    video_id = str(row["video_id"]).strip()

    keypoint_path = find_keypoint_file(row)

    # --------------------------------------------------------
    # Missing keypoints
    # --------------------------------------------------------

    if keypoint_path is None:

        rejection_counts["missing_keypoints"] += 1

        continue


    # --------------------------------------------------------
    # Load keypoints
    # --------------------------------------------------------

    try:

        keypoints = np.load(
            keypoint_path,
            allow_pickle=False
        )

    except Exception:

        rejection_counts["missing_keypoints"] += 1

        continue


    # --------------------------------------------------------
    # Check shape
    # --------------------------------------------------------

    if keypoints.shape != EXPECTED_SHAPE:

        rejection_counts["wrong_shape"] += 1

        continue


    # --------------------------------------------------------
    # Check finite values
    # --------------------------------------------------------

    if not np.isfinite(keypoints).all():

        rejection_counts["non_finite"] += 1

        continue


    # --------------------------------------------------------
    # Check all-zero sequence
    # --------------------------------------------------------

    if np.allclose(keypoints, 0):

        rejection_counts["all_zero"] += 1

        continue


    # --------------------------------------------------------
    # Check exact duplicate sequence
    # --------------------------------------------------------

    digest = keypoint_hash(keypoints)

    if digest in seen_hashes:

        rejection_counts["duplicate_sequence"] += 1

        continue

    seen_hashes.add(digest)


    # --------------------------------------------------------
    # KEEP RECORD
    # --------------------------------------------------------

    record = row.to_dict()

    record["keypoint_path"] = str(
        keypoint_path.relative_to(PROJECT_ROOT)
    )

    clean_records.append(record)


# ============================================================
# CREATE CLEAN DATAFRAME
# ============================================================

clean_df = pd.DataFrame(
    clean_records
)


# ============================================================
# CLEANING RESULT
# ============================================================

print()
print("=" * 75)
print("CLEANING RESULT")
print("=" * 75)

print()
print(
    "Candidate records:",
    len(candidate_records)
)

print(
    "Clean records:",
    len(clean_df)
)

print()
print("Rejected records:")

for reason, count in rejection_counts.items():

    print(
        f"  {reason}: {count}"
    )


# ============================================================
# SIGN COVERAGE
# ============================================================

if len(clean_df) > 0:

    sign_counts = (
        clean_df
        .groupby("gloss")
        .size()
        .reset_index(
            name="clean_clips"
        )
    )

else:

    sign_counts = pd.DataFrame(
        columns=[
            "gloss",
            "clean_clips"
        ]
    )


# ============================================================
# MINIMUM CLIP REQUIREMENT
# ============================================================

sign_counts["meets_15"] = (
    sign_counts["clean_clips"]
    >= MIN_CLIPS_PER_SIGN
)


sign_counts = sign_counts.sort_values(
    [
        "meets_15",
        "clean_clips",
        "gloss"
    ],
    ascending=[
        False,
        False,
        True
    ]
)


# ============================================================
# ELIGIBLE / SHORTFALL SIGNS
# ============================================================

eligible_signs = (
    sign_counts[
        sign_counts["meets_15"]
    ]["gloss"]
    .tolist()
)


short_signs = (
    sign_counts[
        ~sign_counts["meets_15"]
    ]["gloss"]
    .tolist()
)


# ============================================================
# PRINT SIGN COVERAGE
# ============================================================

print()
print("=" * 75)
print("SIGN COVERAGE")
print("=" * 75)

print()
print(
    "Signs with >= 15 clean clips:",
    len(eligible_signs)
)

print(
    "Signs with < 15 clean clips:",
    len(short_signs)
)


# ============================================================
# ELIGIBLE SIGNS
# ============================================================

print()
print("Eligible signs:")

for sign in eligible_signs:

    count = int(
        sign_counts.loc[
            sign_counts["gloss"] == sign,
            "clean_clips"
        ].iloc[0]
    )

    print(
        f"  {sign}: {count}"
    )


# ============================================================
# SHORTFALL SIGNS
# ============================================================

print()
print("Shortfall signs:")

for sign in short_signs:

    count = int(
        sign_counts.loc[
            sign_counts["gloss"] == sign,
            "clean_clips"
        ].iloc[0]
    )

    print(
        f"  {sign}: {count}"
    )


# ============================================================
# SAVE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DATASET.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SAVE CLEAN DATASET
# ============================================================

clean_df.to_csv(
    OUTPUT_DATASET,
    index=False
)


# ============================================================
# SAVE SIGN SUMMARY
# ============================================================

sign_counts.to_csv(
    OUTPUT_SUMMARY,
    index=False
)


# ============================================================
# SAVE REPORT
# ============================================================

report = {
    "candidate_sign_count": len(candidate_glosses),

    "candidate_record_count": len(
        candidate_records
    ),

    "clean_record_count": len(
        clean_df
    ),

    "rejection_counts": rejection_counts,

    "eligible_sign_count": len(
        eligible_signs
    ),

    "shortfall_sign_count": len(
        short_signs
    ),

    "minimum_clips_per_sign": MIN_CLIPS_PER_SIGN,

    "expected_keypoint_shape": list(
        EXPECTED_SHAPE
    ),

    "eligible_signs": eligible_signs,

    "shortfall_signs": short_signs,
}


with open(
    OUTPUT_REPORT,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        report,
        file,
        indent=2
    )


# ============================================================
# FINAL MESSAGE
# ============================================================

print()
print("=" * 75)
print("FINAL CLEANING COMPLETE")
print("=" * 75)

print()
print("Clean dataset:")
print(OUTPUT_DATASET)

print()
print("Sign summary:")
print(OUTPUT_SUMMARY)

print()
print("Cleaning report:")
print(OUTPUT_REPORT)

print()
print("No original files were deleted.")
print("No raw videos were modified.")

print()
print("=" * 75)
print("NEXT STEP: REVIEW SIGN COVERAGE")
print("=" * 75)