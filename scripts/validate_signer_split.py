import os
import numpy as np
import pandas as pd

base = "data/metadata"
files = [
    "train_signer_disjoint.csv",
    "validation_signer_disjoint.csv",
    "test_signer_disjoint.csv",
]

total = 0
all_labels = set()
missing = []
invalid = []

for filename in files:
    df = pd.read_csv(os.path.join(base, filename))
    total += len(df)
    all_labels.update(df["gloss"].astype(str).unique())

    print(f"{filename}: rows={len(df)}, signs={df['gloss'].nunique()}")

    for path in df["keypoint_path"]:
        path = str(path)

        if not os.path.isfile(path):
            missing.append(path)
            continue

        try:
            arr = np.load(path, allow_pickle=False)
            if arr.shape != (30, 1662) or not np.isfinite(arr).all():
                invalid.append((path, arr.shape))
        except Exception as exc:
            invalid.append((path, str(exc)))

print("\nTotal rows:", total)
print("Unique labels across splits:", len(all_labels))
print("Missing files:", len(missing))
print("Invalid arrays:", len(invalid))

if missing:
    print("Example missing paths:", missing[:3])
if invalid:
    print("Example invalid arrays:", invalid[:3])
