import json
import subprocess
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse


BASE = Path("data")
METADATA = BASE / "metadata"
RAW = BASE / "raw"

VOCAB_FILE = METADATA / "vocab_candidates_120.json"

MSASL_DIR = METADATA / "msasl"
MSASL_SPLITS = ["train", "val", "test"]

OUTPUT_DIR = RAW / "msasl"
LOG_FILE = RAW / "download_log.json"


# ---------------------------------------------------------
# First controlled test batch
# ---------------------------------------------------------

MAX_VIDEOS = 20


# ---------------------------------------------------------
# FFmpeg location
# ---------------------------------------------------------

FFMPEG_BIN = (
    Path.home()
    / "AppData"
    / "Local"
    / "Microsoft"
    / "WinGet"
    / "Packages"
    / "Gyan.FFmpeg.Shared_Microsoft.Winget.Source_8wekyb3d8bbwe"
    / "ffmpeg-9.0.1-full_build-shared"
    / "bin"
)


# ---------------------------------------------------------
# Get YouTube video ID
# ---------------------------------------------------------

def get_video_id(url):
    """Extract the YouTube video ID."""
    try:
        query = parse_qs(urlparse(url).query)
        return query.get("v", [None])[0]
    except Exception:
        return None


# ---------------------------------------------------------
# Load candidate vocabulary
# ---------------------------------------------------------

with open(VOCAB_FILE, "r", encoding="utf-8") as f:
    candidates = json.load(f)

candidate_glosses = {
    item["gloss"]
    for item in candidates
}


# ---------------------------------------------------------
# Load MS-ASL class mapping
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
# Collect matching MS-ASL records
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

        records.append(
            {
                "dataset": "msasl",
                "split": split,
                "gloss": gloss,
                "label": label,
                "signer_id": record.get("signer_id"),
                "start": record.get("start"),
                "end": record.get("end"),
                "url": url,
                "video_id": get_video_id(url),
            }
        )


# ---------------------------------------------------------
# Remove duplicate video IDs
# ---------------------------------------------------------

unique_records = []
seen = set()

for record in records:

    video_id = record["video_id"]

    if not video_id:
        continue

    if video_id in seen:
        continue

    seen.add(video_id)
    unique_records.append(record)


# ---------------------------------------------------------
# Deterministic ordering
# ---------------------------------------------------------

unique_records.sort(
    key=lambda x: (
        x["gloss"],
        x["split"],
        x["video_id"],
    )
)


# ---------------------------------------------------------
# Take only the first controlled test batch
# ---------------------------------------------------------

records = unique_records[:MAX_VIDEOS]


# ---------------------------------------------------------
# Prepare output directory
# ---------------------------------------------------------

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------------------------------
# Start a fresh download-test log
# ---------------------------------------------------------

# The previous 20 attempts failed because the wrong Python
# interpreter was used. They were not actual video failures.
#
# Therefore, we intentionally start a fresh log here.

log = []


# ---------------------------------------------------------
# Check FFmpeg location
# ---------------------------------------------------------

ffmpeg_executable = FFMPEG_BIN / "ffmpeg.exe"

if not ffmpeg_executable.exists():

    print("ERROR: FFmpeg executable was not found.")
    print()
    print("Expected location:")
    print(FFMPEG_BIN)
    print()

    sys.exit(1)


# ---------------------------------------------------------
# Print download information
# ---------------------------------------------------------

print("=" * 80)
print("CONTROLLED MS-ASL DOWNLOAD")
print("=" * 80)

print(f"Python executable: {sys.executable}")
print(f"yt-dlp module: python -m yt_dlp")
print(f"FFmpeg directory: {FFMPEG_BIN}")
print()
print(f"Candidate signs: {len(candidate_glosses)}")
print(f"Available matching records: {len(unique_records)}")
print(f"Test batch size: {len(records)}")
print()


# ---------------------------------------------------------
# Download videos
# ---------------------------------------------------------

for index, record in enumerate(records, start=1):

    video_id = record["video_id"]

    output_file = (
        OUTPUT_DIR
        / f"{record['gloss']}__"
          f"{record['split']}__"
          f"{video_id}.%(ext)s"
    )

    print(
        f"[{index}/{len(records)}] "
        f"{record['gloss']} | "
        f"{record['split']} | "
        f"{video_id}"
    )

    # IMPORTANT:
    # sys.executable guarantees that yt-dlp is run using
    # the same Python interpreter as this project/venv.

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
        "--ffmpeg-location",
        str(FFMPEG_BIN),
        "-o",
        str(output_file),
        record["url"],
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    success = result.returncode == 0

    log.append(
        {
            **record,
            "success": success,
            "return_code": result.returncode,
        }
    )

    if success:

        print("  SUCCESS")

    else:

        print("  FAILED")

        # Print only the last part of the error so the terminal
        # does not become unnecessarily large.

        error_text = result.stderr.strip()

        if error_text:
            print(error_text[-1000:])


# ---------------------------------------------------------
# Save download log
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Final report
# ---------------------------------------------------------

print()
print("=" * 80)
print("DOWNLOAD TEST FINISHED")
print("=" * 80)

successes = sum(
    1
    for item in log
    if item["success"]
)

failures = len(log) - successes

print(f"Logged records: {len(log)}")
print(f"Successful downloads: {successes}")
print(f"Failed downloads: {failures}")
print(f"Log: {LOG_FILE}")