import csv
import sys
from pathlib import Path


# =========================================================
# PROJECT ROOT
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =========================================================
# PATHS
# =========================================================

METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
KEYPOINT_DIR = PROJECT_ROOT / "data" / "keypoints"

OUTPUT_FILE = (
    METADATA_DIR / "master_dataset.csv"
)


# =========================================================
# METADATA MANIFESTS
# =========================================================

MANIFEST_FILES = [
    METADATA_DIR / "combined_expanded_manifest.csv",
    METADATA_DIR / "combined_download_manifest.csv",
    METADATA_DIR / "balanced_batch_02_manifest.csv",
    METADATA_DIR / "remaining_manifest.csv",
    METADATA_DIR / "recovery_manifest.csv",
]


# =========================================================
# KEYPOINT DIRECTORIES
# =========================================================

KEYPOINT_DIRS = [
    KEYPOINT_DIR / "expanded_batch_01",
    KEYPOINT_DIR / "balanced_batch_02",
    KEYPOINT_DIR / "remaining_downloads",
    KEYPOINT_DIR / "recovery",
]


# =========================================================
# EXCLUDED ORPHAN RECORD
# =========================================================

# This keypoint exists, but its video ID is not present
# in the available WLASL/MS-ASL metadata.
#
# It is excluded because the project requires metadata
# such as gloss, signer and dataset provenance.

EXCLUDED_VIDEO_IDS = {
    "N5ZUQXL4"
}


# =========================================================
# EXTRACT VIDEO ID
# =========================================================

def get_video_id(file_path):

    parts = file_path.stem.split("__")

    # Normal format:
    # dataset__gloss__split__video_id.npy

    if len(parts) >= 4:
        return parts[3].strip()

    # Fallback format:
    # dataset__gloss__video_id.npy

    if len(parts) == 3:
        return parts[2].strip()

    return None


# =========================================================
# START
# =========================================================

print("=" * 75)
print("BUILD MASTER DATASET")
print("=" * 75)


# =========================================================
# LOAD METADATA
# =========================================================

print()
print("Loading metadata...")

metadata_by_video = {}


for manifest_file in MANIFEST_FILES:

    if not manifest_file.exists():

        print(
            f"WARNING: Missing manifest: "
            f"{manifest_file.name}"
        )

        continue


    with open(
        manifest_file,
        "r",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            video_id = str(
                row.get(
                    "video_id",
                    ""
                )
            ).strip()


            if not video_id:
                continue


            if video_id not in metadata_by_video:

                metadata_by_video[
                    video_id
                ] = row


print(
    f"Unique metadata records: "
    f"{len(metadata_by_video)}"
)


# =========================================================
# FIND ALL KEYPOINT FILES
# =========================================================

print()
print("Loading keypoint files...")

keypoint_records = {}

unexpected_files = []


for directory in KEYPOINT_DIRS:

    if not directory.exists():

        print(
            f"WARNING: Missing keypoint directory: "
            f"{directory}"
        )

        continue


    for file_path in directory.glob("*.npy"):

        video_id = get_video_id(
            file_path
        )


        if not video_id:

            unexpected_files.append(
                file_path
            )

            continue


        if video_id not in keypoint_records:

            keypoint_records[
                video_id
            ] = file_path


print(
    f"Unique keypoint records: "
    f"{len(keypoint_records)}"
)

print(
    f"Unexpected keypoint filenames: "
    f"{len(unexpected_files)}"
)


# =========================================================
# MATCH KEYPOINTS WITH METADATA
# =========================================================

print()
print("Matching metadata and keypoints...")


master_records = []
missing_metadata = []


for video_id, keypoint_path in sorted(
    keypoint_records.items()
):

    # -----------------------------------------------------
    # Exclude orphan keypoint
    # -----------------------------------------------------

    if video_id in EXCLUDED_VIDEO_IDS:

        continue


    # -----------------------------------------------------
    # Find metadata
    # -----------------------------------------------------

    metadata = metadata_by_video.get(
        video_id
    )


    if metadata is None:

        missing_metadata.append(
            video_id
        )

        continue


    # -----------------------------------------------------
    # Create master record
    # -----------------------------------------------------

    master_records.append({

        "dataset":
            metadata.get(
                "dataset",
                ""
            ),

        "split":
            metadata.get(
                "split",
                ""
            ),

        "gloss":
            metadata.get(
                "gloss",
                ""
            ).lower().strip(),

        "label":
            metadata.get(
                "label",
                ""
            ),

        "signer_id":
            metadata.get(
                "signer_id",
                ""
            ),

        "start":
            metadata.get(
                "start",
                ""
            ),

        "end":
            metadata.get(
                "end",
                ""
            ),

        "video_id":
            video_id,

        "keypoint_path":
            str(
                keypoint_path.relative_to(
                    PROJECT_ROOT
                )
            ),

    })


# =========================================================
# SAVE MASTER DATASET
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
    "video_id",
    "keypoint_path",
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
        master_records
    )


# =========================================================
# SUMMARY
# =========================================================

print()
print("=" * 75)
print("MASTER DATASET SUMMARY")
print("=" * 75)

print()

print(
    f"Metadata records: "
    f"{len(metadata_by_video)}"
)

print(
    f"Keypoint records: "
    f"{len(keypoint_records)}"
)

print(
    f"Excluded orphan records: "
    f"{len(EXCLUDED_VIDEO_IDS)}"
)

print(
    f"Master dataset records: "
    f"{len(master_records)}"
)

print(
    f"Missing metadata: "
    f"{len(missing_metadata)}"
)

print(
    f"Unexpected filenames: "
    f"{len(unexpected_files)}"
)


# =========================================================
# DISPLAY ANY UNEXPECTED PROBLEMS
# =========================================================

if missing_metadata:

    print()
    print("Missing metadata video IDs:")

    for video_id in missing_metadata[:20]:

        print(
            f"  {video_id}"
        )


if unexpected_files:

    print()
    print("Unexpected keypoint filenames:")

    for path in unexpected_files[:20]:

        print(
            f"  {path.name}"
        )


# =========================================================
# OUTPUT
# =========================================================

print()
print("Saved to:")
print(OUTPUT_FILE)

print()
print("=" * 75)


# =========================================================
# FINAL VALIDATION
# =========================================================

print("=" * 75)

expected_keypoints = 2265
expected_master = 2264
expected_orphans = 1

if (
    len(keypoint_records) == expected_keypoints
    and len(EXCLUDED_VIDEO_IDS) == expected_orphans
    and len(master_records) == expected_master
    and len(unexpected_files) == 0
):

    print(
        "MASTER DATASET VALIDATION: PASS"
    )

else:

    print(
        "MASTER DATASET VALIDATION: REVIEW REQUIRED"
    )

print("=" * 75)