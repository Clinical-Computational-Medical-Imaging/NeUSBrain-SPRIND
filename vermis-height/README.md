# vermis-height

Automatic measurement of the **cerebellar vermis height** (and AP diameter) on
2D midsagittal segmentations, e.g. from fetal or neonatal ultrasound.

You give it label maps in which the vermis is segmented. It returns a CSV with
one measurement per image, plus a QC image showing where the measurement lines
were drawn.

## Method: chord through the centroid

1. **Clean the mask.** Take the vermis label (default value `2`), keep the
   largest connected component and fill holes inside it.
2. **Convert to millimetres.** Map every pixel to world coordinates (mm) using
   the NIfTI affine or the DICOM pixel spacing. Anisotropic, transposed or
   flipped pixel grids therefore all give the same result.
3. **Find the vermis axis.** Compute the centroid of the vermis and its
   principal axis (PCA). The axis is limited to ±20° from the image vertical,
   because the free PCA axis can tip over when the vermis is almost round. This
   also matches how the vermis height is usually measured by hand.
4. **Measure.**
   - **Height** = length of the straight line through the centroid along that
     axis, from where it enters the vermis to where it leaves it, like a
     caliper line drawn by hand through the middle of the vermis.
   - **AP diameter** = the same line, perpendicular to the height.

Unlike a "bounding box along the axis" measurement, the chord is not inflated by
bumps or irregularities anywhere on the contour.

## Installation

```bash
git clone <this repository>
cd vermis-height
pip install -r requirements.txt
```

Requires Python ≥ 3.9. Compressed DICOMs (JPEG etc.) may also need
`pip install pylibjpeg pylibjpeg-libjpeg`.

## Usage

### 1. Label maps only

The pixel spacing is read from the label files themselves.

```bash
python vermis_height.py --input path/to/labels/ --output results/
```

### 2. Label maps + original image DICOMs (recommended for ultrasound)

The pixel spacing and the QC background are taken from the matching image
DICOM. The label's own spacing is ignored, because segmentation tools often do
not store it correctly.

```bash
python vermis_height.py --input path/to/labels/ --images path/to/dicoms/ --output results/
```

A label map is matched to an image when its file name, without extension and
without a trailing `_seg`, equals either

- the image's file name without extension, e.g. `case01_seg.nii.gz` ↔ `case01.dcm`, or
- the image's DICOM `SeriesDescription`.

The label map must have the same size as the image. A transposed label is
accepted and turned back automatically.

### Input formats

| Input        | Supported                                                                    |
|--------------|------------------------------------------------------------------------------|
| `--input`    | single file or folder (searched recursively): `.nii`, `.nii.gz`, DICOM       |
| `--images`   | folder (searched recursively) of DICOMs; DICOM SEG objects are skipped       |

Label maps must be a **single 2D slice**. Singleton dimensions and an RGB
channel axis are handled automatically.

### Options

| Option               | Default | Meaning                                                                   |
|----------------------|---------|---------------------------------------------------------------------------|
| `--label`            | `2`     | label value of the vermis in the segmentation                             |
| `--max-axis-angle`   | `20`    | max. angle (°) between measurement axis and image vertical; `-1` = no limit |
| `--vertical-axis`    | `0`     | array axis that runs top–bottom in the image (`0` = rows)                 |
| `--no-qc`            | off     | do not write QC images                                                    |

## Output

```
results/
├── measurements.csv
└── qc/
    ├── case01.png
    └── ...
```

### `measurements.csv`

| Column                              | Meaning                                                                 |
|-------------------------------------|-------------------------------------------------------------------------|
| `sample_id`                         | file name of the label map without extension / `_seg`                  |
| `label_path`, `image_path`          | input files used                                                        |
| `success`, `reason`                 | whether the measurement worked, and why not                             |
| `height_mm`                         | **vermis height** (chord through the centroid)                          |
| `ap_mm`                             | AP diameter (perpendicular chord through the centroid)                  |
| `area_mm2`                          | area of the cleaned vermis mask                                         |
| `elongation`                        | `height_mm / ap_mm`                                                     |
| `pca_angle_deg`                     | angle of the unclamped principal axis from the image vertical           |
| `n_components`, `kept_fraction`     | number of mask fragments; fraction of the mask kept as the largest one  |
| `spacing_row_mm`, `spacing_col_mm`  | pixel spacing used                                                      |
| `flags`                             | quality warnings, see below                                             |

### Flags

| Flag                   | Meaning                                                                          |
|------------------------|----------------------------------------------------------------------------------|
| `multiple_components`  | the segmentation had several fragments; only the largest was measured            |
| `axis_unstable`        | elongation < 1.2: the vermis is almost round, so its axis is poorly defined      |
| `axis_clamped`         | the principal axis was more than `--max-axis-angle` from vertical and was limited |
| `spacing_uncalibrated` | spacing is 1 × 1, so the file probably has no calibration and values are in pixels |

Rows with flags should be checked in the QC images.

### QC images

- red line: height
- blue line: AP diameter
- yellow dot: centroid
- without `--images`: kept vermis in light gray, discarded fragments in dark gray
- with `--images`: the image with the vermis contour in green and discarded fragments in orange

The QC image is drawn in array (row, col) layout. It can therefore appear
transposed or flipped compared to a DICOM viewer. The measurement itself is not
affected by that.
