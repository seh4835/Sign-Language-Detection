# Sign Language Recognition

## Project overview

This repository contains research and demonstration pipelines for **100-class isolated American Sign Language (ASL) gloss recognition**. It does not perform continuous-sign recognition or translate ASL sentences. The project uses WLASL and MS-ASL source material, with canonical gloss-to-index mapping and signer-disjoint data manifests.

There are several separate inference paths:

| Path | Features and model | Entry point |
|---|---|---|
| Original webcam Transformer | MediaPipe Holistic output, selected 258 pose-and-hand features over 30 frames, Transformer classifier | webcam_test.py |
| YOLO11 + MediaPipe Hands LSTM | 30 × 180 feature sequence, two-layer LSTM | webcam_yolo_pose_hands_lstm.py |
| Selected final-evaluation baseline | 30 × 180 sequence summarized to 360 mean/standard-deviation values, Logistic Regression | webcam_yolo_pose_hands_temporal_baseline.py and FastAPI |

These are distinct pipelines; their validation results are not a controlled architecture comparison. See the [Model Card](docs/MODEL_CARD.md) and [Data Card](docs/DATA_CARD.md).

## Installation and setup

The repository workflows target Python 3.11. From the project root:

    python -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt

For the YOLO webcam and full development scripts, install the additional direct imports that are not pinned in requirements.txt:

    python -m pip install pandas ultralytics mlflow

These extra packages are currently unpinned in the repository, so this is not a fully locked development environment. The API-only environment can instead install requirements-api.txt. It pins different scikit-learn and PyTorch versions from requirements.txt; verify serialized artifact compatibility in the target environment.

The webcam paths also require their corresponding local checkpoints and artifacts. The temporal-baseline webcam script defaults to outputs/yolo11_pose_test/yolo11n-pose.pt and models/yolo_pose_hands_lstm_v2/webcam_baseline_artifacts/.

## Run the webcam application

The selected final-evaluation baseline webcam application is:

    python webcam_yolo_pose_hands_temporal_baseline.py

Press S to start capturing a sign interval, Space or Enter to predict and record the actual label when experiment mode is enabled, R to reset, and Q or Esc to quit. This app uses the default camera index 0 and requires at least 30 captured frames for a prediction. To show live lighting statistics without running predictions:

    python webcam_yolo_pose_hands_temporal_baseline.py --lighting-calibration

The experimental LSTM webcam path is:

    python webcam_yolo_pose_hands_lstm.py

The original MediaPipe Holistic/Transformer webcam path is:

    python webcam_test.py

The legacy Transformer app appends prediction metadata to reports/webcam_predictions.csv. None of these live paths stores raw webcam frames or videos. The separate YOLO smoke-test section below operates on an input video file and writes derived sample outputs.

## Run the FastAPI API

The API accepts already-extracted numeric features, not raw webcam images or video. A request must contain exactly 30 frames with 180 finite values per frame. The application exposes GET /health and POST /predict; FastAPI provides an interactive schema at /docs.

For a local API environment, install the API requirements and launch from the repository root:

    python -m pip install -r requirements-api.txt
    python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000

Check readiness with:

    curl.exe http://localhost:8000/health

Use /docs to submit a POST /predict request matching the 30 × 180 sequence schema. The API loads the temporal-baseline classifier, normalizer, and 100-class label mapping from models/yolo_pose_hands_lstm_v2/webcam_baseline_artifacts/.

## Docker and Docker Hub

Build and run the API container from the repository root:

    docker build -t sehers/sign-language-api:latest .
    docker run --rm -p 8000:8000 sehers/sign-language-api:latest

The container serves port 8000. Test /health and open http://localhost:8000/docs. To publish the requested tag, authenticate and push:

    docker login
    docker push sehers/sign-language-api:latest

To pull and run a published image:

    docker pull sehers/sign-language-api:latest
    docker run --rm -p 8000:8000 sehers/sign-language-api:latest

These commands follow the checked-in Dockerfile and API port configuration; they were not executed during this documentation update. The current Docker Hub publication status of sehers/sign-language-api:latest was not verified. The API requirements pin scikit-learn 1.7.2 while the full project requirements pin 1.9.0; confirm the serialized classifier loads and predicts in the target image before deployment.

## Evaluation results and limitations

The selected final model is the temporal mean/variance Logistic Regression baseline, not the LSTM or Transformer. The existing [final held-out evaluation report](models/yolo_pose_hands_lstm_v2/final_test_evaluation/final_test_report.md) states it was evaluated once on 248 test samples after fitting on 1,181 training samples, with no validation samples used for fitting. Reported test metrics are:

| Metric | Reported result |
|---|---:|
| Accuracy | 0.379032 |
| Macro precision | 0.406746 |
| Macro recall | 0.405000 |
| Macro-F1 | 0.361831 |
| Weighted-F1 | 0.355363 |
| Predicted classes | 86/100 |
| Correctly recognized classes | 65/100 |

These are report values; no held-out data was re-opened or re-evaluated for this documentation. The 0.80 accuracy and 0.92 top-5 values in configs/config.yaml are targets, not achieved results. The report does not support an 80% accuracy claim or 30 FPS claim.

Validation artifacts report the original Transformer at 0.37751004016064255 Top-1 and 0.6947791164658634 Top-5 (249 examples), the augmented Transformer at 0.24899598393574296 and 0.5742971887550201 (249), and the YOLO/Hands v2 LSTM at 0.271967 accuracy and 0.259095 macro-F1 (239). These runs use different pipelines and sample counts, so treat them as context rather than a controlled model ranking. See the [validation comparison](reports/graphs/validation_model_comparison.csv) and [v2 model-selection analysis](reports/yolo11_migration/v2_model_selection_analysis.md).

The separate reduced lighting study involved one participant and 82 attempts, not a demographic fairness evaluation. Its CSV-based accuracy was 33/82 overall; Bright 9/21, Normal 8/20, Dim 11/20, and Uneven 5/21. Results are preliminary and not statistically conclusive. See the [Responsible AI report](reports/fairness/responsible_ai_report.md) and lighting documents below.

## Privacy

Live webcam frames are processed in memory and are not stored as raw images or videos. The legacy webcam application stores prediction metadata, and the lighting experiment logger stores scalar lighting measurements and sign labels. Offline dataset extraction and smoke-test workflows have separate persistent source-video, keypoint, and derived-output folders. API requests contain feature arrays rather than raw imagery; server-side retention depends on the deployment environment.

## Reproducible checks and CI

The project CI workflow at .github/workflows/ci.yml runs on pushes and pull requests to main using Python 3.11. It runs API, temporal-baseline, cross-validation, and webcam-privacy tests plus a validation-only smoke-training gate. A local command matching the test portion is:

    python -m pytest -q tests/test_api.py tests/test_temporal_baseline.py tests/test_cross_validation.py tests/test_webcam_privacy.py
    python scripts/ci_smoke_train.py

The checks use development data or synthetic smoke fixtures; they do not require re-evaluating the held-out test split. The final held-out evaluation report is the source for final test metrics; do not use the test split for tuning or repeated model selection. Vocabulary provenance is less certain: the repository candidate-selection and label-mapping utilities read test-split label metadata, and the chronology of final vocabulary selection relative to the split is not documented. This prevents certifying that class-list selection was independent of test labels, even though the documented cross-validation and Optuna workflows use train and validation only.

## MLflow, Optuna, and Evidently

MLflow records are stored in the local mlruns directory. The repository’s import script logs existing validation comparison, cross-validation, and weight-decay results:

    python scripts/log_mlflow_experiments.py
    python -m mlflow ui --backend-store-uri ./mlruns

Open http://localhost:5000 to browse the local tracking store. MLflow is imported by project scripts but is not pinned in requirements.txt; install it separately as shown above.

The Optuna tuning script runs three trials across five signer-grouped development folds built from train and validation manifests only:

    python scripts/optuna_tune_transformer.py

It writes reports/optuna/best_trial.json and refuses to overwrite an existing result. Preserve the current result; the command will stop if that path already exists.

The Evidently drift script compares keypoint summaries from train and validation manifests and writes HTML/JSON reports under reports/drift/:

    python scripts/generate_keypoint_drift_report.py

It refuses to overwrite existing drift reports. This monitoring run uses development splits, not the held-out test split.

## YOLO11n-Pose smoke test

The bounded pose smoke test operates on a supplied ASL video and is separate from live webcam inference. Example using the existing train-tagged sample path:

    python scripts/test_yolo11_pose.py --video data/raw/balanced_batch_02/msasl__brown__train__HGBcts4shKw.mp4 --max-frames 30

Add --display to show annotated frames. The script writes a downloaded or used pose model, annotated sample video/frames, keypoint sample, and summary under outputs/yolo11_pose_test/. Do not treat these files as live webcam captures.

## Five-fold signer-aware cross-validation

Cross-validation uses only train_signer_disjoint.csv and validation_signer_disjoint.csv. It groups by dataset::signer_id and applies StratifiedGroupKFold with gloss stratification. The test manifest is not used for fold creation, tuning, early stopping, or model selection.

Create and validate development fold assignments, then run cross-validation:

    python scripts/create_signer_split.py
    python scripts/cross_validate_transformer.py

Fold assignments, class counts, coverage limitations, epoch histories, per-class metrics, and fold-specific checkpoints are written under reports/cross_validation/. Each fold fits normalization on its training partition only. Existing model checkpoints are not modified by this workflow.

## Project documentation

- [Model Card](docs/MODEL_CARD.md)
- [Data Card](docs/DATA_CARD.md)
- [Responsible AI report](reports/fairness/responsible_ai_report.md)
- [Lighting experiment protocol](reports/fairness/lighting_experiment_protocol.md)
- [Lighting experiment analysis](reports/fairness/lighting_experiment_analysis.md)

## Responsible use and open limitations

This is a research prototype with limited closed-set accuracy and uneven class performance. The lighting experiment has one participant, does not include skin-tone labels, and cannot establish demographic fairness or population-level generalization. Dataset licensing and per-record consent are not fully documented in this repository; verify upstream terms before redistribution or new use. Do not use the model as a sole basis for consequential decisions or claim that unachieved targets have been met.

