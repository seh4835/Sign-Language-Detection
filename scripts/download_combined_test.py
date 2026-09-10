import csv
import json
import subprocess
import sys
from pathlib import Path


# =========================================================
# PATHS
# =========================================================

MANIFEST = Path(
    "data/metadata/combined_download_manifest.csv"
)

OUTPUT_DIR = Path(
    "data/raw/combined_test"
)

LOG_FILE = Path(
    "data/raw/combined_download_test_log.json"
)


# =========================================================
# SETTINGS
# =========================================================

MAX_VIDEOS = 100


# =========================================================
# LOAD MANIFEST
# =========================================================

with open(
    MANIFEST,
    "r",
    encoding="utf-8"
) as f:

    reader = csv.DictReader(f)

    records = list(reader)


print("=" * 80)
print("CONTROLLED COMBINED DATASET DOWNLOAD TEST")
print("=" * 80)

print(
    f"Manifest records: {len(records)}"
)


# =========================================================
# SELECT REPRESENTATIVE RECORDS
# =========================================================

# Take records across the manifest rather than
# only downloading the first 100.

if len(records) > MAX_VIDEOS:

    step = len(records) / MAX_VIDEOS

    selected = [
        records[
            int(i * step)
        ]
        for i in range(MAX_VIDEOS)
    ]

else:

    selected = records


print(
    f"Test videos: {len(selected)}"
)

print()


# =========================================================
# CREATE OUTPUT DIRECTORY
# =========================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# LOAD EXISTING LOG
# =========================================================

if LOG_FILE.exists():

    with open(
        LOG_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        log = json.load(f)

else:

    log = []


# =========================================================
# DOWNLOAD
# =========================================================

successful = 0
failed = 0


for index, record in enumerate(
    selected,
    start=1
):

    dataset = record["dataset"]
    gloss = record["gloss"]
    split = record["split"]
    video_id = record["video_id"]
    url = record["url"]

    filename = (
        f"{dataset}__"
        f"{gloss}__"
        f"{split}__"
        f"{video_id}.%(ext)s"
    )

    output_file = (
        OUTPUT_DIR / filename
    )

    print(
        f"[{index}/{len(selected)}] "
        f"{dataset} | "
        f"{gloss} | "
        f"{split} | "
        f"{video_id}"
    )

    command = [

        sys.executable,

        "-m",
        "yt_dlp",

        "-f",
        "mp4[height<=480]/mp4",

        "--no-playlist",

        "--retries",
        "2",

        "--fragment-retries",
        "2",

        "--socket-timeout",
        "30",

        "--no-warnings",

        "-o",
        str(output_file),

        url

    ]


    try:

        result = subprocess.run(
            command,
            capture_output=True,
            text=True
        )

        success = (
            result.returncode == 0
        )

    except Exception as e:

        success = False
        result = None

        error_text = str(e)


    if success:

        successful += 1

        print(
            "    SUCCESS"
        )

        error_text = ""

    else:

        failed += 1

        print(
            "    FAILED"
        )

        if result is not None:

            error_text = (
                result.stderr[-1000:]
            )

            print(
                error_text
            )


    log.append({

        **record,

        "success": success,

        "error": error_text

    })


# =========================================================
# SAVE LOG
# =========================================================

LOG_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    LOG_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        log,
        f,
        indent=2,
        ensure_ascii=False
    )


# =========================================================
# SUMMARY
# =========================================================

print()
print("=" * 80)
print("COMBINED DOWNLOAD TEST FINISHED")
print("=" * 80)

print(
    f"Videos attempted: {len(selected)}"
)

print(
    f"Successful: {successful}"
)

print(
    f"Failed: {failed}"
)

print(
    f"Log: {LOG_FILE}"
)

print(
    f"Output: {OUTPUT_DIR}"
)