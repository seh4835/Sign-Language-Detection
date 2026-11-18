"""Audit label, split, duplicate, and keypoint invariants."""

from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "data" / "metadata"
SPLITS = (
    "train_signer_disjoint.csv",
    "validation_signer_disjoint.csv",
    "test_signer_disjoint.csv",
)


def main():
    mapping = json.loads(
        (META / "label_mapping.json").read_text(encoding="utf-8")
    )
    frames = {name: pd.read_csv(META / name) for name in SPLITS}
    report = {
        "mapping_classes": len(mapping),
        "splits": {},
        "overlap": {},
        "invalid_keypoints": [],
    }

    for name, frame in frames.items():
        groups = (
            frame["dataset"].astype(str)
            + "::"
            + frame["signer_id"].astype(str)
        )
        report["splits"][name] = {
            "rows": len(frame),
            "classes": int(frame["gloss"].nunique()),
            "signer_groups": int(groups.nunique()),
            "video_ids": int(frame["video_id"].nunique()),
            "keypoint_paths": int(frame["keypoint_path"].nunique()),
            "source_split_column": frame["split"].value_counts(
                dropna=False
            ).to_dict(),
        }
        for path in frame["keypoint_path"]:
            keypoint_path = ROOT / str(path)
            if not keypoint_path.is_file():
                report["invalid_keypoints"].append(
                    {"path": str(path), "error": "missing"}
                )
                continue
            array = np.load(keypoint_path, allow_pickle=False)
            if array.shape != (30, 1662) or not np.isfinite(array).all():
                report["invalid_keypoints"].append(
                    {"path": str(path), "shape": list(array.shape)}
                )

    for index, first_name in enumerate(SPLITS):
        first = frames[first_name]
        first_groups = set(
            first["dataset"].astype(str)
            + "::"
            + first["signer_id"].astype(str)
        )
        for second_name in SPLITS[index + 1 :]:
            second = frames[second_name]
            second_groups = set(
                second["dataset"].astype(str)
                + "::"
                + second["signer_id"].astype(str)
            )
            report["overlap"][f"{first_name} vs {second_name}"] = {
                "dataset_signer": len(first_groups & second_groups),
                "video_id": len(
                    set(first["video_id"]) & set(second["video_id"])
                ),
                "keypoint_path": len(
                    set(first["keypoint_path"])
                    & set(second["keypoint_path"])
                ),
            }

    output = ROOT / "reports" / "data_pipeline_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved audit report to {output}")


if __name__ == "__main__":
    main()
