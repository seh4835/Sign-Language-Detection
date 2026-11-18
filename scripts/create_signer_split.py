"""Create and validate five signer-aware development folds.

Only the existing training and validation metadata are read. The held-out test
metadata is intentionally not opened or modified by this script.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cross_validation import (
    N_FOLDS,
    RANDOM_STATE,
    create_fold_assignments,
    load_development_data,
)


ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "data" / "metadata"
REPORTS = ROOT / "reports" / "cross_validation"


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    development = load_development_data(
        META / "train_signer_disjoint.csv",
        META / "validation_signer_disjoint.csv",
    )
    assignments, report = create_fold_assignments(
        development, n_splits=N_FOLDS, random_state=RANDOM_STATE
    )
    assignments.to_csv(REPORTS / "fold_assignments.csv", index=False)
    (REPORTS / "fold_validation.json").write_text(
        json.dumps(
            {
                "n_folds": N_FOLDS,
                "random_state": RANDOM_STATE,
                "development_examples": len(assignments),
                "development_groups": int(assignments["group_id"].nunique()),
                "folds": report,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    for item in report:
        missing = item["missing_validation_classes"]
        warning = f"; missing validation classes: {missing}" if missing else ""
        print(
            f"Fold {item['fold']}: train={item['training_examples']} "
            f"validation={item['validation_examples']} "
            f"groups={item['training_groups']}/{item['validation_groups']}"
            f"{warning}"
        )
    print(f"Saved fold assignments and validation report under {REPORTS}")


if __name__ == "__main__":
    main()
