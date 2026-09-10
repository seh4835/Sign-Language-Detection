import sys
from pathlib import Path

# Add the project root to Python's import path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from pathlib import Path

from src.keypoints.extract import extract_video_keypoints


VIDEO = Path(
    "data/raw/msasl/again__train__bDjiz9Ngcfg.mp4"
)

OUTPUT = Path(
    "data/keypoints/msasl/test_sequence.npy"
)


print("=" * 70)
print("KEYPOINT EXTRACTION TEST")
print("=" * 70)

sequence = extract_video_keypoints(
    VIDEO,
    OUTPUT,
    sequence_length=30,
)

print()
print("Video:", VIDEO)
print("Output:", OUTPUT)
print("Shape:", sequence.shape)
print("Data type:", sequence.dtype)
print("Finite values:", np.isfinite(sequence).all())
print("Minimum:", sequence.min())
print("Maximum:", sequence.max())
print("File exists:", OUTPUT.exists())

print()

if sequence.shape != (30, 1662):
    raise AssertionError(
        f"Unexpected shape: {sequence.shape}"
    )

if not np.isfinite(sequence).all():
    raise AssertionError(
        "Keypoints contain NaN or infinite values."
    )

if not OUTPUT.exists():
    raise AssertionError(
        "Output .npy file was not created."
    )

print("RANDOM TEST CHECK: PASS")
print("Keypoint extraction test completed successfully.")