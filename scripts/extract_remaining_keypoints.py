import sys
from pathlib import Path

import numpy as np


# =========================================================
# PROJECT ROOT
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =========================================================
# IMPORT EXTRACTOR
# =========================================================

from src.keypoints.extract import extract_video_keypoints


# =========================================================
# PATHS
# =========================================================

VIDEO_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "remaining_downloads"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "keypoints"
    / "remaining_downloads"
)


VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm"
}


# =========================================================
# MAIN
# =========================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    videos = sorted(
        p
        for p in VIDEO_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower()
        in VIDEO_EXTENSIONS
    )

    print("=" * 70)
    print("REMAINING DATA KEYPOINT EXTRACTION")
    print("=" * 70)

    print()
    print(
        f"Videos found: {len(videos)}"
    )

    print(
        f"Output directory: {OUTPUT_DIR}"
    )

    print()

    successful = 0
    failed = 0


    # =====================================================
    # PROCESS
    # =====================================================

    for i, video_path in enumerate(
        videos,
        start=1
    ):

        print(
            f"[{i:03d}/{len(videos)}] "
            f"Processing: {video_path.name}"
        )

        try:

            output_path = (
                OUTPUT_DIR
                / f"{video_path.stem}.npy"
            )

            extract_video_keypoints(
                str(video_path),
                str(output_path),
                sequence_length=30
            )

            keypoints = np.load(
                output_path
            )

            # Validate basic output
            assert (
                keypoints.shape
                == (30, 1662)
            )

            assert (
                keypoints.dtype
                == np.float32
            )

            assert np.isfinite(
                keypoints
            ).all()

            assert (
                keypoints.std() > 0
            )

            print(
                f"       PASS | "
                f"Shape={keypoints.shape} | "
                f"Dtype={keypoints.dtype}"
            )

            successful += 1

        except Exception as e:

            print(
                f"       FAIL | {e}"
            )

            failed += 1


    # =====================================================
    # SUMMARY
    # =====================================================

    print()
    print("=" * 70)
    print("REMAINING KEYPOINT EXTRACTION SUMMARY")
    print("=" * 70)

    print(
        f"Videos found: {len(videos)}"
    )

    print(
        f"Successful:   {successful}"
    )

    print(
        f"Failed:       {failed}"
    )

    print(
        f"Output:       {OUTPUT_DIR}"
    )

    print()

    if failed == 0:

        print(
            "ALL REMAINING KEYPOINT EXTRACTIONS "
            "COMPLETED SUCCESSFULLY."
        )

    else:

        print(
            "SOME KEYPOINT EXTRACTIONS FAILED. "
            "REVIEW THE OUTPUT ABOVE."
        )

    print("=" * 70)


if __name__ == "__main__":
    main()