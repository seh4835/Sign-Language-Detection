import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.keypoints.extract import extract_video_keypoints


VIDEO_DIR = PROJECT_ROOT / "data" / "raw" / "combined_test"
OUTPUT_DIR = PROJECT_ROOT / "data" / "keypoints" / "combined_test"

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    videos = sorted(
        p for p in VIDEO_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )

    print("=" * 70)
    print("COMBINED TEST KEYPOINT EXTRACTION")
    print("=" * 70)

    print(f"Videos found: {len(videos)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    successful = 0
    failed = 0

    for i, video_path in enumerate(videos, 1):
        print(f"[{i:02d}/{len(videos)}] Processing: {video_path.name}")

        try:
            output_name = video_path.stem + ".npy"
            output_path = OUTPUT_DIR / output_name

            extract_video_keypoints(
                str(video_path),
                str(output_path),
                sequence_length=30
            )

            keypoints = np.load(output_path)

            print(
                f"       PASS | Shape={keypoints.shape} | "
                f"Dtype={keypoints.dtype}"
            )

            successful += 1

        except Exception as e:
            print(f"       FAIL | {e}")
            failed += 1

    print()
    print("=" * 70)
    print("COMBINED KEYPOINT EXTRACTION SUMMARY")
    print("=" * 70)

    print(f"Videos found: {len(videos)}")
    print(f"Successful:   {successful}")
    print(f"Failed:       {failed}")
    print(f"Output:       {OUTPUT_DIR}")

    if failed == 0:
        print()
        print("ALL COMBINED TEST VIDEOS EXTRACTED SUCCESSFULLY.")
    else:
        print()
        print("SOME VIDEOS FAILED EXTRACTION. REVIEW THE OUTPUT ABOVE.")


if __name__ == "__main__":
    main()