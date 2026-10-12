# NeUSBrain-SPRIND

NeUSBrain – 3D brain norms from newborn ultrasound.

This repository contains our submission for automatically measuring the
**vermis height** in **mid-sagittal neonatal cranial ultrasound (NeUS)
images**. The pipeline has three stages:

1. **Classify**: find usable images and the mid-sagittal plane (CNN classifiers).
2. **Segment**: segment the skull and the cerebellum/vermis (nnU-Net v2, 2D).
3. **Measure**: fit a statistical shape model (SSM) to the vermis
   segmentation and measure the height between two corresponding landmarks.

```
 DICOM / NIfTI / PNG
        │
        ▼
 ┌──────────────────────┐   models/classifiers/incl_excl/cp.keras
 │ 1a include / exclude │   EfficientNet, 288×256 input
 └──────────┬───────────┘
            ▼
 ┌──────────────────────┐   models/classifiers/cor_sag/cp.keras
 │ 1b coronal/sagittal  │   (+ optional 3-class mid-sagittal model)
 └──────────┬───────────┘
            ▼
 ┌──────────────────────┐   nnU-Net 2D: skull model + cerebellum model
 │ 2  segmentation      │   → labelmap: 1 = skull, 2 = cerebellum/vermis
 └──────────┬───────────┘
            ▼
 ┌──────────────────────┐   PCA shape model (100 corresponding landmarks)
 │ 3a SSM fit (ASM)     │   + HD95 quality check
 └──────────┬───────────┘
            ▼
 ┌──────────────────────┐   distance landmark 15 ↔ 60, 5 mirrored pairs,
 │ 3b vermis height [mm]│   trimmed mean, verticality check
 └──────────────────────┘
```

---

## Repository structure

```
.
├── models/
│   ├── classifiers/
│   │   ├── incl_excl/cp.keras        # include / exclude classifier (TensorFlow)
│   │   └── cor_sag/cp.keras          # coronal / sagittal classifier (TensorFlow)
│   ├── nnunet/                       # nnU-Net results folders go here (see below)
│   └── ssm/                          # PCA shape model (pca_shape_model.npz) goes here
├── requirements/                     # one requirements file per environment
├── src/
│   ├── classification/
│   │   ├── classify.py               # stage 1: classify any DICOM/NIfTI/PNG folder
│   │   └── predict_incl_excl_dicom.py# alternative: DICOM-only, with US region cropping
│   ├── segmentation/
│   │   ├── segment.py                # stage 2: nnU-Net inference on classified cases
│   │   ├── nnunet_inference.py       # nnU-Net inference + Dice/HD95 on a test set
│   │   └── make_vermis_labelmap.py   # merge predictions → 1 = skull, 2 = cerebellum
│   └── vermis_height/
│       ├── 01_preprocess_data.py     # DICOM + labels → cropped 2D NIfTI with mm spacing
│       ├── 02_correspondent_landmarking.py  # CPD point correspondence on label-2 border
│       ├── 03_build_pca.py           # GPA + PCA shape model
│       ├── 04_fit_pca.py             # fit shape model to a (predicted) segmentation
│       ├── 05_visualize_landmarks.py # show landmark indices on images
│       ├── 06_measure_vermis.py      # vermis height from fitted landmarks
│       ├── 07_create_vermis_statistics.py # growth curves vs. corrected age
│       └── run_pipeline.sh           # runs 01 → 07
└── training/
    ├── classification/               # labeling tool + plane classifier training
    └── segmentation/                 # build nnU-Net datasets (skull, cerebellum)
```

## Installation

The classifiers use TensorFlow and the segmentation uses PyTorch/nnU-Net.
Use **separate environments** so the two frameworks do not conflict
(CUDA/cuDNN versions):

```bash
# 1) classification
conda create -n neus_tf python=3.11 -y && conda activate neus_tf
pip install -r requirements/classification.txt

# 2) segmentation
conda create -n neus_nnunet python=3.11 -y && conda activate neus_nnunet
pip install -r requirements/segmentation.txt

# 3) vermis height (can also be installed into the nnU-Net environment)
pip install -r requirements/vermis_height.txt
```

## Models

| Stage | Model | Location | Input | Output |
|---|---|---|---|---|
| 1a | Include / exclude CNN (Keras, EfficientNet) | `models/classifiers/incl_excl/cp.keras` | RGB 288×256 | `exclude`, `include` |
| 1b | Coronal / sagittal CNN (Keras) | `models/classifiers/cor_sag/cp.keras` | RGB 288×256 | `Coronal`, `Saggital` |
| 1c | Mid-sagittal plane CNN (optional, Keras EfficientNet-B3) | train with `training/classification/train_plane_classifier_tf.py` | RGB 300×300 | `not_relevant`, `mid_saggital`, `relevant_coronal` |
| 2a | Skull segmentation (nnU-Net v2, `2d`) | `models/nnunet/Dataset009_*/nnUNetTrainer__nnUNetPlans__2d/` | US image | 1 = skull |
| 2b | Cerebellum segmentation (nnU-Net v2, `2d`) | `models/nnunet/Dataset010_midsagittal_cerebellum/nnUNetTrainer__nnUNetPlans__2d/` | mid-sagittal US image | 1 = cerebellum |
| 3 | Sagittal PCA shape model | `models/ssm/sagittal/pca_shape_model.npz` | labelmap (label 2) | 100 fitted landmarks |

> The nnU-Net results folders and the PCA shape model are too large or
> data-derived to keep in git. Copy them into `models/nnunet/` and
> `models/ssm/` as shown above. You need the complete nnU-Net folder
> (`dataset.json`, `plans.json`, `fold_*/checkpoint_final.pth`).
> To rebuild them, see [Training](#training).

---

## 1. Classification

`src/classification/classify.py` accepts any folder structure containing DICOM
series, NIfTI files or PNG/JPG images. For every case it:

1. converts the case to a persistent `.nii.gz` in `--harmonized-dir`
   (needed by the segmentation step),
2. takes a representative 2D slice, converts it to RGB uint8, resizes it to
   288×256 and runs the Keras classifier (on CPU),
3. writes one row per case to `--output-csv` straight away (a crash keeps all
   results up to that point).

Output columns: `case_id, kind, status, error, nifti_path, original_path,
classifier_label, classifier_confidence, included`.

### 1a. Include / exclude

Removes images that cannot be used (wrong anatomy, Doppler, annotations, bad
quality, ...).

```bash
conda activate neus_tf
python src/classification/classify.py \
    --input /path/to/images \
    --tf-model models/classifiers/incl_excl/cp.keras \
    --harmonized-dir outputs/harmonized \
    --output-csv outputs/incl_excl.csv \
    --min-confidence 0.5
```

### 1b. Plane: coronal vs. sagittal

Run the same script with the plane model. `--class-names` gives the model's
output order and `--positive-class` the class that sets `included=True`:

```bash
python src/classification/classify.py \
    --input /path/to/images \
    --tf-model models/classifiers/cor_sag/cp.keras \
    --class-names Coronal Saggital \
    --positive-class Saggital \
    --harmonized-dir outputs/harmonized \
    --output-csv outputs/cor_sag.csv
```

Combine both results. Only images that are **included and sagittal** go to
segmentation:

```bash
python - <<'EOF'
import pandas as pd
a = pd.read_csv("outputs/incl_excl.csv")
b = pd.read_csv("outputs/cor_sag.csv")[["case_id", "classifier_label", "included"]]
m = a.merge(b, on="case_id", suffixes=("", "_plane"))
m["included"] = m["included"] & m["included_plane"]
m.to_csv("outputs/mid_sagittal_cases.csv", index=False)
print(m["included"].value_counts())
EOF
```

### 1c. Mid-sagittal (optional)

The coronal/sagittal model does not separate mid-sagittal from para-sagittal
slices. For that we trained a 3-class EfficientNet
(`not_relevant` / `mid_saggital` / `relevant_coronal`) on reviewer labels from
`training/classification/label_review_tool.py`. See
[Training → Classifiers](#classifiers).

---

## 2. Segmentation

Two 2D nnU-Net v2 models are used:

* **Skull model**: label 1 = skull.
* **Cerebellum model** (`Dataset010_midsagittal_cerebellum`): trained only on
  mid-sagittal images, binary mask of the cerebellum/vermis (label 2 of the
  original annotations).

`src/segmentation/segment.py` reads the classifier CSV, keeps the
rows with `included == True` and runs nnU-Net on their harmonized NIfTI.
Each batch runs in its own subprocess, so a crash (e.g. out of memory) only
loses that batch. Cases whose spacing would make nnU-Net resample them to a
very large size are skipped and flagged in the manifest.

```bash
conda activate neus_nnunet

# cerebellum / vermis
python src/segmentation/segment.py \
    --classifier-csv outputs/mid_sagittal_cases.csv \
    --nnunet-model-folder models/nnunet/Dataset010_midsagittal_cerebellum/nnUNetTrainer__nnUNetPlans__2d \
    --nnunet-folds all \
    --device cuda \
    --output-dir outputs/seg_cerebellum \
    --manifest-out outputs/seg_cerebellum_manifest.csv

# skull (optional, only used for visualization)
python src/segmentation/segment.py \
    --classifier-csv outputs/mid_sagittal_cases.csv \
    --nnunet-model-folder models/nnunet/Dataset009_Tuechert_impr_neus/nnUNetTrainer__nnUNetPlans__2d \
    --nnunet-folds all \
    --device cuda \
    --output-dir outputs/seg_skull \
    --manifest-out outputs/seg_skull_manifest.csv
```

Use `--device cpu` or `--device mps` (Apple Silicon) without a CUDA GPU.

### Merge into a vermis labelmap

The shape model expects **label 2 = cerebellum**. Merge the predictions
(spacing and affine are kept):

```bash
python src/segmentation/make_vermis_labelmap.py \
    --cerebellum-dir outputs/seg_cerebellum \
    --skull-dir outputs/seg_skull \
    --output-dir outputs/labelmaps \
    --largest-component
```

`--largest-component` removes small false-positive islands, which would
otherwise distort the contour used for the shape fit.

### Evaluate a segmentation model

`src/segmentation/nnunet_inference.py` runs a trained model on a test set
(`imagesTs`) and reports Dice / HD95 against `labelsTs`, with overlay images:

```bash
export nnUNet_raw=/path/to/nnUNet_raw
export nnUNet_results=/path/to/nnUNet_results
python src/segmentation/nnunet_inference.py --dataset 010 --folds all
```

---

## 3. Vermis height measurement

The vermis height is **not** measured directly on the raw mask border,
because CNN border pixels are noisy. We fit a **statistical shape model** to
the vermis contour first and then measure between two fixed, anatomically
corresponding landmarks.

### Method

1. **Contour extraction**: the border of label 2 (cerebellum/vermis) is
   extracted and resampled to a dense point set (400 points).
2. **Point correspondence** (`02`): a 100-point template is registered to every
   training contour, first with rigid and then with deformable *Coherent Point
   Drift* (CPD, `pycpd`). Every landmark index then marks the same anatomical
   position in every image. Cases with a large CPD residual are removed
   (median + 3·MAD, or a fixed limit). This also removes images in the wrong
   plane.
3. **Shape model** (`03`): Generalized Procrustes Analysis removes
   translation, rotation and scale, then PCA gives a mean shape and the main
   modes of variation (all modes are stored; the number needed for 95 % of
   the variance is reported):

   `shape = mean + Σ bᵢ · √λᵢ · modeᵢ`

4. **Model fit** (`04`): for a new (predicted) segmentation, the mean shape is
   placed on its contour by CPD and refined with an *Active Shape Model*
   loop: pose (rotation, scale, translation) and shape parameters `b` are
   updated alternately, and `b` is clamped to ±3 SD so the result stays a
   plausible vermis. Fit quality is the **HD95 (mm)** between the fitted shape
   and the segmentation border. Fits above `--max-hd95-mm` are flagged
   `fit_quality_ok = False`.
5. **Measurement** (`06`): landmarks **15** and **60** of the 100-point
   sagittal model mark the two ends of the vermis height axis. They were
   chosen visually with `05_visualize_landmarks.py`. A single landmark pair
   can be slightly off near a curved tip, so we measure **5 mirrored pairs**

   `(15+k, 60−k), k = −2 … 2`

   convert each distance to mm with the image's pixel spacing, and report
   `mean_mm`, `median_mm`, `trimmed_mean_mm` (min and max dropped; **our
   final value**), `std_mm`, `min_mm`, `max_mm`. Pairs whose line deviates
   more than `--max-angle-deviation-deg` (25°) from vertical are discarded;
   if none are left, the case is marked `measurement_reliable = False`.

### Measuring new images (inference)

Starting from the labelmaps of stage 2:

```bash
conda activate neus_nnunet   # or any env with requirements/vermis_height.txt

# fit the shape model
python src/vermis_height/04_fit_pca.py \
    --pca-model models/ssm/sagittal/pca_shape_model.npz \
    --input outputs/labelmaps \
    --output-dir outputs/fitted \
    --max-hd95-mm 1.5

# measure the vermis height
python src/vermis_height/06_measure_vermis.py \
    --fitted-landmarks-dir outputs/fitted/fitted_landmarks \
    --fit-results-csv outputs/fitted/fit_results.csv \
    --spacing-dir outputs/labelmaps \
    --labels-dir outputs/labelmaps \
    --images-dir outputs/harmonized \
    --output-dir outputs/measurements \
    --landmark-a 15 --landmark-b 60 \
    --max-angle-deviation-deg 25
```

Outputs in `outputs/measurements/`:

| File | Content |
|---|---|
| `measurements_summary.csv` | one row per image: `trimmed_mean_mm` (vermis height), mean/median/std/min/max, `measurement_reliable` |
| `measurement_detail/<id>.csv` | the 5 candidate distances per image |
| `visualizations/<id>.png` | image + border + fitted landmarks + measurement lines |
| `measurement_failures.csv` | images that could not be measured and why |

The pixel spacing comes from the NIfTI header. DICOM input
gives real mm values (ultrasound region calibration or `PixelSpacing`).
**PNG input has no spacing; it is stored as 1 mm/pixel, so results are
in pixels, not mm.**

### Building the shape model (training)

Steps 01–03 need ground-truth segmentations (label 2 = cerebellum) and a CSV
with plane labels (`image_path,user_label`). `run_pipeline.sh` runs the
complete chain (01 → 04 → 06 → 07) on a labeled dataset:

```bash
bash src/vermis_height/run_pipeline.sh \
    /path/to/dicom_root \
    /path/to/segmentations \
    /path/to/plane_labels.csv \
    outputs/ssm_work
cp -r outputs/ssm_work/pca/sagittal models/ssm/
```

| Step | Script | Main output |
|---|---|---|
| 01 | `01_preprocess_data.py` | `images/` and `labels/` per plane: cropped to the US region, 224×224, mm spacing in the affine; `matched_pairs.csv` |
| 02 | `02_correspondent_landmarking.py --n-landmarks 100 --cpd-alpha 4.0` | `correspondence/<plane>/landmarks_all_samples.csv`, quality report |
| 03 | `03_build_pca.py` | `pca/<plane>/pca_shape_model.npz`, explained variance, shape scores |
| 04 | `04_fit_pca.py` | fitted landmarks + HD95 (on the training labels, for validation) |
| 05 | `05_visualize_landmarks.py` | numbered landmarks on images (for choosing landmark indices) |
| 06 | `06_measure_vermis.py` | vermis height per image |
| 07 | `07_create_vermis_statistics.py` | growth curves of vermis height vs. corrected (post-menstrual) age |

If you change `--n-landmarks` or rebuild the model, check the landmark
indices again with `05_visualize_landmarks.py`, because they depend on the
template.

---

## Training

### Classifiers

* `training/classification/label_review_tool.py`: Tkinter tool for labeling
  DICOM images as `mid_saggital` / `relevant_coronal` / `not_relevant`
  (keys `1`/`2`/`3`, `n` = not sure). It writes `review_results.csv`.
* `training/classification/train_plane_classifier_tf.py`: EfficientNet
  (B0–B4) classifier with US-region cropping from the DICOM metadata, focal
  loss, class weights and cosine learning rate. Keeps a leaderboard of runs.

  ```bash
  python training/classification/train_plane_classifier_tf.py \
      --csv review_results.csv --output models_out/ \
      --backbone B3 --img_height 300 --img_width 300 --val_split 0.1
  ```
* `training/classification/train_plane_classifier_monai.py`: the same task in
  PyTorch/MONAI.

### Segmentation (nnU-Net v2)

```bash
export nnUNet_raw=/path/to/nnUNet_raw
export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
export nnUNet_results=/path/to/nnUNet_results

# cerebellum dataset: mid-sagittal images only, label 2 → foreground
python training/segmentation/prep_nnunet_cerebellum.py \
    --dcm_root /path/to/dicom_root \
    --seg_root /path/to/segmentations \
    --csv_file review_results.csv \
    --output_dataset $nnUNet_raw/Dataset010_midsagittal_cerebellum \
    --dataset_name Dataset010_midsagittal_cerebellum

nnUNetv2_plan_and_preprocess -d 010 --verify_dataset_integrity
nnUNetv2_train 010 2d all

# skull dataset: extends an existing nnU-Net dataset with new cases
python training/segmentation/prep_nnunet_skull.py \
    --existing_dataset $nnUNet_raw/Dataset003_submission \
    --dcm_root /path/to/dicom_root \
    --seg_root /path/to/segmentations \
    --output_dataset $nnUNet_raw/Dataset009_Tuechert_impr_neus \
    --dataset_name Dataset009_Tuechert_impr_neus \
    --crop_failure full

nnUNetv2_plan_and_preprocess -d 009 --verify_dataset_integrity
nnUNetv2_train 009 2d all
```

Copy the results folders (`$nnUNet_results/Dataset0XX_*`) into
`models/nnunet/`.

---

## Notes and limitations

* All models are 2D. For 3D or multi-frame inputs, `classify.py` uses the
  middle slice (`--slice-selection middle`) or the slice with the most
  content (`max-content`).
* The classifiers run on CPU on purpose: TensorFlow XLA fails on clusters
  without a full CUDA toolkit.
* The vermis height is only reliable for true mid-sagittal images. Check
  `fit_quality_ok` (HD95) and `measurement_reliable` (verticality) before
  using a value.
* No patient data is included in this repository.
