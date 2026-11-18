# Data Card: 100-Class Isolated ASL Recognition

## Summary

This project prepares sign-labeled video and keypoint sequences for 100-class isolated American Sign Language (ASL) gloss recognition. Each example is one labeled sign clip. The data does not represent continuous ASL sentences or translation.

## Sources and provenance

The project references two source collections:

- **WLASL**, including WLASL v0.3 metadata at data/metadata/WLASL_v0.3.json.
- **MS-ASL**, including class and split metadata under data/metadata/msasl/.

Source video and local derived data are organized under data/raw/ and data/keypoints/; the YOLO11 + MediaPipe Hands extractor defaults to data/keypoints_yolo_pose_hands/. Exact source records are traceable through the project manifests and video IDs.

The vocabulary selection code builds a candidate pool from glosses shared between WLASL and MS-ASL and uses source-specific minimum counts of 10 WLASL clips and 20 MS-ASL clips for that candidate pool. The project model mapping contains 100 contiguous class indices. The canonical mapping is data/metadata/label_mapping.json. The candidate-pool script is not itself a license or proof that every source record was retained. It reads MS-ASL train, validation, and test metadata while building candidates. The label-mapping utility also reads the three canonical split manifests. The repository does not record whether the final vocabulary was frozen before test-split label metadata was consulted; therefore, independence of vocabulary selection from held-out label metadata cannot be confirmed. This is a metadata-provenance limitation, not a new evaluation of test examples.

## Dataset selection and vocabulary

The model is a closed-set classifier over the 100 glosses in label_mapping.json. Canonical training targets are derived from the gloss through this mapping. The source CSV label field is not a safe substitute for the canonical target; the dataset-readiness report documents source-label inconsistencies and directs users to map glosses through the canonical mapping.

The canonical split manifests are:

- data/metadata/train_signer_disjoint.csv
- data/metadata/validation_signer_disjoint.csv
- data/metadata/test_signer_disjoint.csv

Do not combine or regenerate these casually. The project readiness audit reports no overlap of dataset/signer groups across the three manifests. Five-fold development cross-validation uses only the train and validation manifests, groups records by dataset::signer_id, and applies StratifiedGroupKFold stratified by gloss. For the documented five-fold cross-validation and Optuna workflows, the held-out test split is not used for fold creation or tuning. However, the vocabulary utilities described above read test-split label metadata; the chronology of final vocabulary selection relative to the split is undocumented.

## Preprocessing and feature extraction

The repository contains more than one feature pipeline:

1. **YOLO11 pose + MediaPipe Hands pipeline:** Each frame has 177 continuous values: 17 body keypoints × (x, y, confidence), and 21 landmarks × (x, y, z) for each hand. Three binary indicators record body, left-hand, and right-hand presence. The extractor samples 30 frames uniformly across a clip and writes raw (30, 177), presence (30, 3), and combined (30, 180) arrays. Body x/y coordinates are normalized by frame dimensions; hand z values are clipped; missing feature groups are zero-filled. Feature normalization is fit on training data only and preserves the presence indicators.
2. **Original MediaPipe Holistic pipeline:** The per-frame representation has 1,662 values (face, pose, and both hands). The original Transformer uses the 258-value pose-and-hand slice and uniformly samples 30 frames, with normalization statistics stored in its checkpoint.

The extraction code stores derived keypoint/features for offline training. Inference pipelines use the feature contract that matches their own model artifact; the 180- and 258-feature representations are not interchangeable.

## Data quality and class balance

The project’s data-readiness and model-selection reports describe variable class counts and sparse per-class validation support across 100 classes. Some classes have few examples, which makes per-class metrics unstable. The v2 model-selection report notes classes with zero correct validation predictions. These data and support limitations constrain generalization and should be shown alongside aggregate metrics.

The project also performs content-safe exclusions for the v2 training view and records exclusions separately. Preserve those records when reproducing the run. Do not silently substitute a source label for the gloss mapping or silently move samples between signer splits.

## Licensing, consent, and use limitations

The repository identifies WLASL and MS-ASL as the source collections but does not establish the license, redistribution permission, or consent status for each source video. Review the applicable upstream dataset terms and source-specific restrictions before copying, publishing, or using any video or derived artifact outside the permitted scope. This card does not assert that all records are freely redistributable or that separate consent was obtained for every person depicted.

## Privacy safeguards

Offline dataset preparation stores downloaded source videos and derived keypoint arrays in project data directories. Restrict access and retention according to upstream terms and the project’s purpose.

For live webcam inference, frames and extracted features are processed in memory and raw webcam frames/videos are not stored by the live capture paths. The legacy webcam app may append prediction metadata, while the lighting logger stores scalar lighting measurements and sign labels. The FastAPI prediction endpoint accepts a 30 × 180 numerical feature sequence, not raw video or images. Server-side logging and retention are deployment-specific and are not established by the repository.

## Representation gaps and known limitations

- No skin-tone demographic labels are available in the project data, and no multi-participant demographic evaluation was conducted.
- The single-participant lighting study is preliminary and is not demographic fairness evidence.
- Dataset clips and signers do not cover all ASL users, regional variants, ages, signing styles, camera types, environments, or accessibility needs.
- Dataset collection and annotation practices are inherited from the source datasets; the repository does not provide complete per-record annotator, license, or consent provenance.
- The 100-label mapping limits predictions to the selected vocabulary and does not describe unrestricted ASL.

## References within the project

- [Canonical label mapping](../data/metadata/label_mapping.json)
- [Dataset readiness and signer-split audit](../reports/yolo11_migration/dataset_readiness.md)
- [Model Card](MODEL_CARD.md)
- [Responsible AI report](../reports/fairness/responsible_ai_report.md)

