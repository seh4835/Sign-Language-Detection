# Final Held-Out Test Evaluation

The selected temporal mean/variance baseline was fit once using only the corrected derived training view and evaluated once on the untouched corrected test view.

## Protocol

- Train samples used for fitting: **1181**
- Validation samples used for fitting: **0**
- Test samples evaluated exactly once: **248**
- Algorithm: `LogisticRegression(max_iter=500, solver="lbfgs", random_state=42)`
- Features: sequence mean concatenated with sequence standard deviation (360 dimensions)
- Normalization: `FeatureNormalizer` fitted on training data only

## Validation Comparison

| Metric | Validation | Final test |
|---|---:|---:|
| Accuracy | 0.343096 | 0.379032 |
| Macro precision | 0.324857 | 0.406746 |
| Macro recall | 0.364167 | 0.405000 |
| Macro-F1 | 0.305771 | 0.361831 |
| Weighted-F1 | not saved | 0.355363 |

## Final Test Results

- Accuracy: **0.379032**
- Macro precision: **0.406746**
- Macro recall: **0.405000**
- Macro-F1: **0.361831**
- Weighted-F1: **0.355363**
- Predicted classes: **86/100**
- Correctly recognized classes: **65/100**

Per-class metrics are in `test_per_class_metrics.csv`; the complete confusion matrix is in `test_confusion_matrix.csv`.
