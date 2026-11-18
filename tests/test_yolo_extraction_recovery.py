from pathlib import Path

import numpy as np
import pytest

from scripts.extract_yolo_pose_hands_dataset import (
    index_videos,
    is_source_resolution_failure,
    resolve_video_candidates,
    save_npz_without_overwrite,
    source_content_conflicts,
)


def test_raw_video_index_preserves_leading_underscore_id(tmp_path: Path) -> None:
    raw = tmp_path / "data" / "raw" / "recovery"
    raw.mkdir(parents=True)
    video = raw / "msasl__sorry___LGqHR3FhL8.mp4"
    video.write_bytes(b"video")

    index = index_videos(tmp_path, {"_LGqHR3FhL8", "LGqHR3FhL8"})

    assert index["_LGqHR3FhL8"] == [video]
    assert "LGqHR3FhL8" not in index


def test_identical_video_aliases_resolve_to_one_source(tmp_path: Path) -> None:
    first = tmp_path / "one.mp4"
    second = tmp_path / "two.mp4"
    first.write_bytes(b"same source bytes")
    second.write_bytes(b"same source bytes")

    resolved, error, digest = resolve_video_candidates([second, first])

    assert resolved == first
    assert error == ""
    assert digest is not None


def test_different_video_aliases_remain_blocked(tmp_path: Path) -> None:
    first = tmp_path / "one.mp4"
    second = tmp_path / "two.mp4"
    first.write_bytes(b"source one")
    second.write_bytes(b"source two")

    resolved, error, digest = resolve_video_candidates([first, second])

    assert resolved is None
    assert "different content" in error
    assert digest is None


def test_identical_source_already_assigned_to_another_split_is_blocked() -> None:
    digest = "same-video-hash"
    test_key = ("test", "wlasl", "35520")
    validation_key = ("validation", "msasl", "eP816GBFVeg")
    source_hashes = {digest: {test_key}}

    assert source_content_conflicts(source_hashes, digest, validation_key) == [test_key]


@pytest.mark.parametrize(
    "error, expected",
    [
        ("Expected one raw video, found 0.", True),
        ("Expected one raw video, found 2.", True),
        ("Video has only 20 frames; need at least 30.", False),
        ("Could not read sampled frame 66.", False),
    ],
)
def test_safe_source_retry_filter(error: str, expected: bool) -> None:
    assert is_source_resolution_failure(error) is expected


def test_npz_writer_never_overwrites_an_existing_output(tmp_path: Path) -> None:
    destination = tmp_path / "existing.npz"
    destination.write_bytes(b"preserve this exact file")
    before = destination.read_bytes()

    with pytest.raises(FileExistsError):
        save_npz_without_overwrite(
            destination,
            raw=np.zeros((30, 177), dtype=np.float32),
            presence=np.zeros((30, 3), dtype=np.float32),
            lstm=np.zeros((30, 180), dtype=np.float32),
        )

    assert destination.read_bytes() == before
    assert list(tmp_path.glob(".existing.*.npz.tmp")) == []
