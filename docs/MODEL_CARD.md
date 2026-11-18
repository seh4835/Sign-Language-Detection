# Model Card: Sign Language Recognition

## Summary

This repository contains several research and demonstration pipelines for **100-class isolated American Sign Language (ASL) gloss recognition**. An input sample represents one sign, not continuous signing, sign-language translation, or natural-language understanding.

The final held-out evaluation report covers the selected temporal mean/variance Logistic Regression baseline. The repository also contains separate Transformer and LSTM pipelines. These are distinct model and feature-processing paths; their validation numbers should not be read as a controlled head-to-head comparison.

## Intended purpose and users

The system is intended for research, coursework, and controlled demonstrations of isolated-sign classification from pose and hand features. It may help developers explore temporal sampling, keypoint processing, model evaluation, and lighting effects.

It is not validated for production communication, education assessment, medical, legal, emergency, employment, or other high-impact decisions. It does not recognize continuous signing, translate ASL sentences, or reliably detect signs outside its fixed 100-class vocabulary. Users should not rely on a single prediction as authoritative.

## Model and feature pipelines

The repository includes three relevant approaches:

| Pipeline | Input and architecture | Role and evidence |
|---|---|---|
| Original Transformer | MediaPipe Holistic features are represented as 30 frames with 1,662 features per frame; the model selects the 258 pose-and-hand features at indices 1404–1661. A linear projection maps to 128 dimensions, followed by positional encoding, two Transformer encoder layers with four heads and a 256-wide feed-forward layer, mean pooling, and a 100-class output head. | Original and augmented Transformer validation results are recorded in the validation comparison artifact. The legacy webcam entry point is webcam_test.py. |
| YOLO11 + MediaPipe Hands LSTM v2 | Each of 30 frames contains 177 continuous features (17 body pose points and 21 landmarks per hand, each with three values) plus three presence indicators, for a 180-feature sequence. The LSTM has two 128-unit layers, 0.25 recurrent dropout, 0.3 head dropout, and a 100-class linear output. | Experimental model selected and assessed on development validation data; the cited model-selection report does not evaluate this LSTM on the held-out test set. |
| Temporal mean/variance baseline | The same 30 × 180 LSTM-ready representation is normalized, then summarized as per-feature sequence means concatenated with standard deviations (360 values). A Logistic Regression classifier with max_iter=500, solver=lbfgs, and random_state=42 predicts one of 100 classes. | This is the model evaluated in the final held-out report and used by the temporal-baseline API artifacts. |

### Validation comparison

The reported validation results are included for context only. They come from different model runs and, for the original Transformer versus the v2 LSTM, different feature pipelines and validation sample counts. They do not establish a controlled ranking across all architectures.

| Model/run | Validation examples | Top-1 / accuracy | Top-5 | Macro-F1 |
|---|---:|---:|---:|---:|
| Original Transformer | 249 | 0.37751004016064255 | 0.6947791164658634 | Not reported in the comparison artifact |
| Augmented Transformer | 249 | 0.24899598393574296 | 0.5742971887550201 | Not reported in the comparison artifact |
| YOLO11 + MediaPipe Hands LSTM v2 | 239 | 0.271967 | Not reported there | 0.259095 |
| Temporal mean/variance Logistic Regression baseline | 239 | 0.343096 | Not reported there | 0.305771 |

Sources: [Transformer validation comparison](../reports/graphs/validation_model_comparison.csv) and [v2 model-selection analysis](../reports/yolo11_migration/v2_model_selection_analysis.md). The final evaluated Logistic Regression baseline is distinct from both the LSTM and Transformer.

## Input and preprocessing

For the YOLO11 + MediaPipe Hands path, the extractor obtains 17 COCO body keypoints and 21 landmarks for each hand. Body values are x, y, and confidence; hand values are x, y, and relative z. Missing groups are zero-filled and accompanied by body/left-hand/right-hand presence indicators. Coordinates are normalized by frame width or height, and hand z values are clipped to the configured range. Sequences are uniformly sampled to 30 frames. The feature normalizer is fit on training sequences only and normalizes detected continuous features while preserving the binary presence indicators.

For the original Transformer path, the project uses MediaPipe Holistic output and selects the pose-and-hand slice from each frame, then samples 30 frames uniformly and applies checkpoint normalization statistics.

The API accepts a numerical sequence of exactly 30 frames × 180 features. It does not accept raw image or video input. A caller must provide compatible preprocessed features.

## Final held-out evaluation

The only held-out metrics stated here are transcribed from the existing [Final Held-Out Test Evaluation report](../models/yolo_pose_hands_lstm_v2/final_test_evaluation/final_test_report.md). That report states the selected temporal mean/variance baseline was fit using 1,181 training samples, with zero validation samples used for fitting, and evaluated once on 248 test samples.

| Metric | Reported final test result |
|---|---:|
| Accuracy | 0.379032 |
| Macro precision | 0.406746 |
| Macro recall | 0.405000 |
| Macro-F1 | 0.361831 |
| Weighted-F1 | 0.355363 |
| Predicted classes | 86/100 |
| Correctly recognized classes | 65/100 |

These results do not meet the 0.80 accuracy target in configs/config.yaml. No test-set metrics are claimed for the LSTM or Transformer in this card.

## Lighting robustness study

The reduced exploratory lighting study recorded 82 attempts for one participant (P01) across Bright, Normal, Dim, and Uneven. The CSV-based analysis reports 33/82 exact-match correct predictions overall (40.24%). By condition: Bright 9/21 (42.86%), Normal 8/20 (40.00%), Dim 11/20 (55.00%), and Uneven 5/21 (23.81%). Bright has one extra attempt; Uneven includes one supplemental attempt. EXP06 contains the actual-label typo ba instead of bad, preserved and scored as an incorrect exact-match record.

This one-participant study is preliminary and not statistically conclusive. It is not demographic fairness evidence and does not support population-level generalization. See the [lighting protocol](../reports/fairness/lighting_experiment_protocol.md), [lighting analysis](../reports/fairness/lighting_experiment_analysis.md), and [Responsible AI report](../reports/fairness/responsible_ai_report.md).

## Limitations and known risks

- The classifier is closed-set: it always selects among its 100 labels, including when the input is unsupported or ambiguous.
- The final reported test accuracy is 0.379032; performance is not reliable enough for unvalidated real-world communication.
- Class-level validation support is sparse and uneven. The model-selection report notes classes with no correct validation predictions and warns that per-class scores based on very few examples are unstable.
- The Transformer, LSTM, and selected temporal baseline use different feature pipelines and runs; validation results are not a controlled architecture experiment.
- Vocabulary provenance is incomplete: repository utilities for candidate selection and label mapping read split label metadata that includes the test split. The record does not establish whether the final 100-class vocabulary was frozen before that metadata was consulted, so independence of label-space selection from test labels cannot be certified.
- The API confidence is a classifier probability output; calibration and abstention behavior are not established by the final report.
- No multi-participant or skin-tone-stratified evaluation was available. No demographic fairness claim is supported.
- Results may change with signer, camera, environment, signing style, framing, occlusion, and lighting.

## Privacy and data handling

The live webcam capture paths process camera frames and keypoints in memory; they do not persist raw webcam frames or video. The legacy Transformer webcam app appends prediction metadata to reports/webcam_predictions.csv, and the lighting experiment logger stores scalar lighting measurements and sign labels. The FastAPI endpoint accepts derived numerical features rather than images or video.

Offline dataset tools do persist source video and derived keypoint artifacts for training and evaluation. Smoke-test utilities may also write annotated sample outputs. These offline artifacts are separate from live webcam capture. Deployment-host logging and retention depend on the operator's environment and are not controlled by this repository.

## API and Docker deployment

The FastAPI application exposes GET /health and POST /predict. The prediction request must contain a sequence shaped 30 × 180; see the API schema and the interactive FastAPI documentation at /docs after launch. The container uses Python 3.11, installs requirements-api.txt, copies the API modules and temporal-baseline artifacts, and serves on port 8000.

From the repository root, these commands follow the checked-in Dockerfile and port configuration; they were not executed during this documentation update:

    docker build -t sehers/sign-language-api:latest .
    docker run --rm -p 8000:8000 sehers/sign-language-api:latest

To publish that tag, authenticate with Docker Hub and run:

    docker login
    docker push sehers/sign-language-api:latest

The intended image tag is sehers/sign-language-api:latest. Whether that tag is currently published on Docker Hub was not verified. The repository pins scikit-learn 1.9.0 in requirements.txt and 1.7.2 in requirements-api.txt; compatibility of the serialized classifier across these versions should be checked in the target container before relying on deployment.



