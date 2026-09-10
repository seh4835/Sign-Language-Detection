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
# IMPORT EXISTING KEYPOINT EXTRACTOR
# =========================================================

from src.keypoints.extract import extract_video_keypoints


# =========================================================
# PATHS
# =========================================================

VIDEO_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "recovery"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "keypoints"
    / "recovery"
)


# =========================================================
# SETTINGS
# =========================================================

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm"
}

EXPECTED_SHAPE = (30, 1662)


# =========================================================
# MAIN
# =========================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    videos = sorted(
        path
        for path in VIDEO_DIR.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in VIDEO_EXTENSIONS
        )
    )

    print("=" * 70)
    print("RECOVERY KEYPOINT EXTRACTION")
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

    for index, video_path in enumerate(
        videos,
        start=1
    ):

        output_path = (
            OUTPUT_DIR
            / f"{video_path.stem}.npy"
        )

        print(
            f"[{index:03d}/{len(videos):03d}] "
            f"Processing: {video_path.name}"
        )

        try:

            # -------------------------------------------------
            # Skip if already extracted
            # -------------------------------------------------

            if output_path.exists():

                try:

                    existing = np.load(
                        output_path
                    )

                    if (
                        existing.shape
                        == EXPECTED_SHAPE
                    ):

                        print(
                            "       SKIP | "
                            "Keypoints already exist"
                        )

                        successful += 1
                        continue

                except Exception:
                    pass


            # -------------------------------------------------
            # Extract using existing extractor
            # -------------------------------------------------

            extract_video_keypoints(
                str(video_path),
                str(output_path),
                sequence_length=30
            )


            # -------------------------------------------------
            # Validate output
            # -------------------------------------------------

            keypoints = np.load(
                output_path
            )

            if keypoints.shape != EXPECTED_SHAPE:

                raise ValueError(
                    f"Unexpected shape: "
                    f"{keypoints.shape}; "
                    f"expected {EXPECTED_SHAPE}"
                )

            if keypoints.dtype != np.float32:

                raise ValueError(
                    f"Unexpected dtype: "
                    f"{keypoints.dtype}; "
                    f"expected float32"
                )

            if not np.isfinite(
                keypoints
            ).all():

                raise ValueError(
                    "Keypoints contain "
                    "NaN or Inf values."
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


    # =========================================================
    # SUMMARY
    # =========================================================

    print()
    print("=" * 70)
    print("RECOVERY KEYPOINT EXTRACTION SUMMARY")
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
            "ALL RECOVERY KEYPOINT "
            "EXTRACTIONS COMPLETED SUCCESSFULLY."
        )

    else:

        print(
            "SOME RECOVERY KEYPOINT "
            "EXTRACTIONS FAILED."
        )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    main()