#!/bin/bash
# Vermis height pipeline (statistical shape model + landmark measurement).
#
# Usage:
#   bash src/vermis_height/run_pipeline.sh <DICOM_ROOT> <LABELS_ROOT> <PLANE_CSV> <WORK_DIR>
#
#   DICOM_ROOT   root folder with the ultrasound DICOM images
#   LABELS_ROOT  root folder with the segmentations (<patient_id>/<image_id>.nii[.gz],
#                label 1 = skull, label 2 = cerebellum/vermis)
#   PLANE_CSV    CSV with columns image_path,user_label (user_label in
#                {mid_saggital, relevant_coronal, not_relevant})
#   WORK_DIR     output folder for all intermediate and final results

set -euo pipefail

if [ "$#" -ne 4 ]; then
    sed -n 2,13p "$0"
    exit 1
fi

DICOM_ROOT="$1"
LABELS_ROOT="$2"
PLANE_CSV="$3"
WORK_DIR="$4"

# Landmark pair that spans the vermis height on the 100-point sagittal model
LANDMARK_A=15
LANDMARK_B=60

cd "$(dirname "$0")"

python3 01_preprocess_data.py \
    --images-dataset "$DICOM_ROOT" \
    --labels-dataset "$LABELS_ROOT" \
    --output-dir "$WORK_DIR" \
    --csv "$PLANE_CSV"

python3 02_correspondent_landmarking.py \
    --data-dir "$WORK_DIR" \
    --output-dir "$WORK_DIR/correspondence" \
    --n-landmarks 100 \
    --target-density 400 \
    --cpd-alpha 4.0

python3 03_build_pca.py \
    --correspondence-dir "$WORK_DIR/correspondence" \
    --output-dir "$WORK_DIR/pca"

python3 04_fit_pca.py \
    --pca-model "$WORK_DIR/pca/sagittal/pca_shape_model.npz" \
    --input "$WORK_DIR/labels/sagittal" \
    --output-dir "$WORK_DIR/fitted/sagittal"

# Optional: check which landmark indices correspond to the vermis extremes
#python3 05_visualize_landmarks.py \
#    --landmarks-dir "$WORK_DIR/fitted/sagittal/fitted_landmarks" \
#    --images-dir "$WORK_DIR/images/sagittal" \
#    --labels-dir "$WORK_DIR/labels/sagittal" \
#    --output-dir "$WORK_DIR/landmark_previews/sagittal" \
#    --n-samples 5

python3 06_measure_vermis.py \
    --fitted-landmarks-dir "$WORK_DIR/fitted/sagittal/fitted_landmarks" \
    --spacing-dir "$WORK_DIR/images/sagittal" \
    --images-dir "$WORK_DIR/images/sagittal" \
    --labels-dir "$WORK_DIR/labels/sagittal" \
    --output-dir "$WORK_DIR/measurements/sagittal" \
    --landmark-a "$LANDMARK_A" --landmark-b "$LANDMARK_B" \
    --max-angle-deviation-deg 25

python3 07_create_vermis_statistics.py \
    --matched-pairs-csv "$WORK_DIR/matched_pairs.csv" \
    --sagittal-measurements "$WORK_DIR/measurements/sagittal/measurements_summary.csv" \
    --output-dir "$WORK_DIR/growth_tracking"
