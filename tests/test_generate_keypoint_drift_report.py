import numpy as np

from scripts.generate_keypoint_drift_report import (
    feature_columns,
    summarize_feature_statistics,
)


def test_statistics_map_to_each_feature_in_expected_order():
    features = np.array(
        [
            [1.0, 10.0],
            [3.0, 14.0],
        ]
    )

    names = feature_columns()[:8]
    values = summarize_feature_statistics(features)

    assert dict(zip(names, values[:8])) == {
        "feature_000_mean": 2.0,
        "feature_000_std": 1.0,
        "feature_000_min": 1.0,
        "feature_000_max": 3.0,
        "feature_001_mean": 12.0,
        "feature_001_std": 2.0,
        "feature_001_min": 10.0,
        "feature_001_max": 14.0,
    }
