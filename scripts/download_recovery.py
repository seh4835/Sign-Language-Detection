import csv
import json
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse, parse_qs


# =========================================================
# PROJECT ROOT
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# =========================================================
# PATHS
# =========================================================

MANIFEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "recovery_manifest.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "recovery"
)

LOG_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "recovery_download_log.json"
)


# =========================================================
# SETTINGS
# =========================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

LOG_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# GET VIDEO ID
# =========================================================

def get_video_id(url):
    """Extract YouTube video ID from a URL."""

    try:
        parsed = urlparse(url)

        query = parse_qs(
            parsed.query
        )

        video_id = query.get(
            "v",
            [None]
        )[0]

        if video_id:
            return video_id

        # Handle youtu.be URLs
        if parsed.netloc.lower() in {
            "youtu.be",
            "www.youtu.be"
        }:

            return parsed.path.strip("/")

    except Exception:
        pass

    return None


# =========================================================
# LOAD MANIFEST
# =========================================================

print("=" * 80)
print("RECOVERY DATASET DOWNLOAD")
print("=" * 80)

print()
print("Loading recovery manifest...")

with open(
    MANIFEST_FILE,
    "r",
    encoding="utf-8"
) as f:

    reader = csv.DictReader(f)
    records = list(reader)


print(
    f"Manifest records: {len(records)}"
)

print(
    f"Output directory: {OUTPUT_DIR}"
)

print()


# =========================================================
# EXISTING VIDEO IDS
# =========================================================

existing_ids = set()

for path in OUTPUT_DIR.glob("*"):

    if not path.is_file():
        continue

    video_id = path.stem

    if video_id:
        existing_ids.add(video_id)


print(
    f"Existing recovery videos: "
    f"{len(existing_ids)}"
)

print()


# =========================================================
# DOWNLOAD
# =========================================================

successful = 0
failed = 0
skipped = 0

log = []


for index, record in enumerate(
    records,
    start=1
):

    url = str(
        record.get(
            "url",
            ""
        )
    ).strip()

    gloss = str(
        record.get(
            "gloss",
            ""
        )
    ).strip()

    dataset = str(
        record.get(
            "dataset",
            ""
        )
    ).strip()

    video_id = str(
        record.get(
            "video_id",
            ""
        )
    ).strip()

    if not video_id:
        video_id = get_video_id(url)

    print(
        f"[{index:04d}/{len(records):04d}] "
        f"{dataset} | {gloss} | {video_id}"
    )


    # -----------------------------------------------------
    # Missing URL / ID
    # -----------------------------------------------------

    if not url or not video_id:

        print(
            "       FAIL | Missing URL or video ID"
        )

        failed += 1

        log.append({
            "dataset": dataset,
            "gloss": gloss,
            "video_id": video_id,
            "url": url,
            "status": "failed",
            "reason": "missing_url_or_video_id"
        })

        continue


    # -----------------------------------------------------
    # Already downloaded
    # -----------------------------------------------------

    if video_id in existing_ids:

        print(
            "       SKIP | Already downloaded"
        )

        skipped += 1

        log.append({
            "dataset": dataset,
            "gloss": gloss,
            "video_id": video_id,
            "url": url,
            "status": "skipped",
            "reason": "already_downloaded"
        })

        continue


    # -----------------------------------------------------
    # Output filename
    # -----------------------------------------------------

    safe_gloss = (
        gloss
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )

    filename = (
        f"{dataset}__"
        f"{safe_gloss}__"
        f"{video_id}.mp4"
    )

    output_path = OUTPUT_DIR / filename


    # -----------------------------------------------------
    # yt-dlp command
    # -----------------------------------------------------

    command = [

        sys.executable,
        "-m",
        "yt_dlp",

        "--no-playlist",

        "--quiet",
        "--no-warnings",

        "--retries",
        "2",

        "--fragment-retries",
        "2",

        "--socket-timeout",
        "15",

        "--restrict-filenames",

        "-f",
        "mp4/best[ext=mp4]/best",

        "-o",
        str(output_path),

        url
    ]


    try:

        result = subprocess.run(
            command,
            capture_output=True,
            text=True
        )


        # -------------------------------------------------
        # Successful download
        # -------------------------------------------------

        if (
            result.returncode == 0
            and output_path.exists()
            and output_path.stat().st_size > 0
        ):

            print(
                "       PASS | Downloaded"
            )

            successful += 1

            existing_ids.add(
                video_id
            )

            log.append({
                "dataset": dataset,
                "gloss": gloss,
                "video_id": video_id,
                "url": url,
                "status": "success",
                "file": str(output_path)
            })


        # -------------------------------------------------
        # Failed download
        # -------------------------------------------------

        else:

            error_message = (
                result.stderr.strip()
                if result.stderr
                else "yt-dlp failed"
            )

            print(
                f"       FAIL | {error_message[:150]}"
            )

            failed += 1

            # Remove incomplete file if present
            if output_path.exists():

                try:
                    output_path.unlink()
                except Exception:
                    pass

            log.append({
                "dataset": dataset,
                "gloss": gloss,
                "video_id": video_id,
                "url": url,
                "status": "failed",
                "reason": error_message[:500]
            })


    except Exception as e:

        print(
            f"       FAIL | {e}"
        )

        failed += 1

        log.append({
            "dataset": dataset,
            "gloss": gloss,
            "video_id": video_id,
            "url": url,
            "status": "failed",
            "reason": str(e)
        })


# =========================================================
# SAVE LOG
# =========================================================

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
# FINAL SUMMARY
# =========================================================

print()
print("=" * 80)
print("RECOVERY DOWNLOAD SUMMARY")
print("=" * 80)

print()
print(
    f"Manifest records: {len(records)}"
)

print(
    f"Successful:       {successful}"
)

print(
    f"Failed:           {failed}"
)

print(
    f"Skipped:          {skipped}"
)

print()

print(
    f"Videos currently in recovery folder: "
    f"{len(list(OUTPUT_DIR.glob('*.mp4')))}"
)

print()

print("Videos saved to:")
print(OUTPUT_DIR)

print()

print("Log saved to:")
print(LOG_FILE)

print()
print("=" * 80)
print("RECOVERY DOWNLOAD COMPLETE")
print("=" * 80)