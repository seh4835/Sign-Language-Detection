import cv2
import numpy as np
from pathlib import Path

from src.keypoints.holistic import HolisticExtractor


def extract_video_keypoints(
    video_path,
    output_path,
    sequence_length=30,
):
    """
    Extract a fixed-length keypoint sequence from a video.

    Instead of running MediaPipe on every frame, this function
    selects `sequence_length` evenly distributed frames first
    and processes only those frames.

    Output shape:
        (sequence_length, 1662)

    Raw video frames are processed in memory only and are never
    saved by this function.
    """

    video_path = Path(video_path)
    output_path = Path(output_path)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if total_frames <= 0:
        cap.release()
        raise RuntimeError(
            f"Video contains no readable frames: {video_path}"
        )

    # Select 30 evenly distributed frame positions.
    if total_frames >= sequence_length:

        frame_indices = np.linspace(
            0,
            total_frames - 1,
            sequence_length,
            dtype=int,
        )

    else:

        # If the video has fewer than 30 frames,
        # process all available frames and later pad.
        frame_indices = np.arange(total_frames)

    extractor = HolisticExtractor()

    frame_features = []

    try:

        for frame_index in frame_indices:

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                int(frame_index),
            )

            success, frame = cap.read()

            if not success:
                continue

            frame_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            features = extractor.process_frame(
                frame_rgb
            )

            frame_features.append(features)

    finally:

        cap.release()
        extractor.close()

    if not frame_features:
        raise RuntimeError(
            f"No readable frames in: {video_path}"
        )

    frame_features = np.asarray(
        frame_features,
        dtype=np.float32,
    )

    # -----------------------------------------------------
    # Ensure exactly sequence_length frames
    # -----------------------------------------------------

    if len(frame_features) >= sequence_length:

        sequence = frame_features[:sequence_length]

    else:

        padding_count = (
            sequence_length - len(frame_features)
        )

        padding = np.repeat(
            frame_features[-1][None, :],
            padding_count,
            axis=0,
        )

        sequence = np.concatenate(
            [
                frame_features,
                padding,
            ],
            axis=0,
        )

    # -----------------------------------------------------
    # Save only keypoints
    # -----------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        output_path,
        sequence,
    )

    return sequence