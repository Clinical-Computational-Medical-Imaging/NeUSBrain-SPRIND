"""
Smoke test for the whole pipeline on SYNTHETIC data (no patient data needed).

Stages (each one is skipped if its dependencies are missing):

    classification  classify.py with both Keras models on synthetic PNG,
                    NIfTI and DICOM inputs -> checks that the models load,
                    every case is classified and the CSV schema is right.
    segmentation    make_vermis_labelmap.py on synthetic predictions, and -
                    only if --nnunet-model is given and nnunetv2 is
                    installed - segment.py with that trained model.
    vermis          synthetic vermis-like masks with a KNOWN height ->
                    02 correspondence -> 03 PCA -> 04 fit -> 06 measure.
                    Checks that every step runs, the fit HD95 is small and
                    the measured height matches the known height.

Usage:
    python tests/smoke_test.py                       # all stages
    python tests/smoke_test.py --stages vermis       # one stage
    python tests/smoke_test.py --nnunet-model models/nnunet/Dataset010_midsagittal_cerebellum/nnUNetTrainer__nnUNetPlans__2d

Exit code 0 = all executed checks passed.
"""

import argparse
import importlib.util
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"

N_VERMIS_SAMPLES = 15
IMAGE_SIZE = 224
SPACING_X_MM = 0.12   # horizontal (columns), deliberately anisotropic
SPACING_Y_MM = 0.10   # vertical (rows)
HEIGHT_TOLERANCE = 0.10   # relative error allowed for the measured vermis height
MAX_HD95_MM = 1.5          # same cutoff as run_pipeline.sh
MIN_MEASURED_FRACTION = 0.8


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

RESULTS = []


def report(stage: str, check: str, status: str, detail: str = "") -> None:
    RESULTS.append((stage, check, status, detail))
    print(f"[{status:4s}] {stage:14s} {check}" + (f"  ({detail})" if detail else ""), flush=True)


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def run(cmd, cwd=None, timeout=3600):
    t0 = time.time()
    proc = subprocess.run([str(c) for c in cmd], cwd=cwd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        print("    ---- stdout (tail) ----\n" + "\n".join(proc.stdout.splitlines()[-30:]))
        print("    ---- stderr (tail) ----\n" + "\n".join(proc.stderr.splitlines()[-30:]))
    return proc.returncode == 0, time.time() - t0


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

def synthetic_us_image(rng: np.random.Generator, h: int = 300, w: int = 260) -> np.ndarray:
    """Speckle-like grayscale image with a bright fan shape, uint8."""
    img = rng.gamma(1.5, 25, size=(h, w))
    yy, xx = np.mgrid[0:h, 0:w]
    fan = (np.abs(xx - w / 2) < (yy + 20) * 0.6)
    img = img * (0.2 + 0.8 * fan)
    return np.clip(img, 0, 255).astype(np.uint8)


def vermis_polygon(cy, cx, half_height, half_width, angle_deg, notch_depth, n=360):
    """Ellipse-like vermis outline with a notch (fastigium) on its anterior
    side, so the shape has an unambiguous orientation for correspondence.
    Returns (N, 2) points as (x=col, y=row) in IMAGE coordinates."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = 1.0 - notch_depth * np.exp(-((np.angle(np.exp(1j * t))) ** 2) / (2 * 0.25 ** 2))
    x = half_width * r * np.cos(t)
    y = half_height * np.sin(t)
    a = np.deg2rad(angle_deg)
    xr = x * np.cos(a) - y * np.sin(a)
    yr = x * np.sin(a) + y * np.cos(a)
    return np.stack([cx + xr, cy + yr], axis=1)


def make_vermis_cases(out_dir: Path, rng: np.random.Generator):
    """Writes labels/sagittal/*.nii.gz and images/sagittal/*.nii.gz with the
    exact layout and affine of 01_preprocess_data.py. Returns {id: height_mm}."""
    import cv2

    prep = load_module(SRC / "vermis_height" / "01_preprocess_data.py", "prep01")
    affine = prep.build_affine(SPACING_X_MM, SPACING_Y_MM)
    truth = {}

    for i in range(N_VERMIS_SAMPLES):
        sample_id = f"synthetic_{i:03d}"
        half_h = rng.uniform(55, 75)          # vertical semi-axis (px)
        half_w = rng.uniform(35, 48)
        angle = rng.uniform(-6, 6)
        cy, cx = IMAGE_SIZE / 2 + rng.uniform(-8, 8), IMAGE_SIZE / 2 + rng.uniform(-8, 8)

        label = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        # skull (label 1): a thick arc at the top of the image
        cv2.ellipse(label, (IMAGE_SIZE // 2, -60), (130, 100), 0, 0, 180, 1, thickness=8)
        poly = vermis_polygon(cy, cx, half_h, half_w, angle, notch_depth=0.35)
        cv2.fillPoly(label, [np.round(poly).astype(np.int32)], 2)

        image = synthetic_us_image(rng, IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)
        image[label == 2] += 60

        prep.save_nifti(label, out_dir / "labels" / "sagittal" / f"{sample_id}.nii.gz", affine, is_label=True)
        prep.save_nifti(image, out_dir / "images" / "sagittal" / f"{sample_id}.nii.gz", affine, is_label=False)

        # Known height: full extent along the long (rotated vertical) axis,
        # in mm. The axis is tilted by `angle`, so both spacings contribute.
        a = np.deg2rad(angle)
        dx_px, dy_px = 2 * half_h * -np.sin(a), 2 * half_h * np.cos(a)
        truth[sample_id] = float(np.hypot(dx_px * SPACING_X_MM, dy_px * SPACING_Y_MM))

    return truth


# ---------------------------------------------------------------------------
# Stage: classification
# ---------------------------------------------------------------------------

def stage_classification(work: Path, rng: np.random.Generator) -> None:
    stage = "classification"
    missing = [m for m in ("tensorflow", "cv2", "nibabel", "pandas", "SimpleITK") if not has_module(m)]
    if missing:
        report(stage, "dependencies", "SKIP", f"missing: {', '.join(missing)}")
        return

    import cv2
    import nibabel as nib
    import pandas as pd
    import SimpleITK as sitk

    inp = work / "cls_input"
    (inp / "dicom_case").mkdir(parents=True)
    for i in range(3):
        cv2.imwrite(str(inp / f"png_case_{i}.png"), synthetic_us_image(rng))
    nib.save(nib.Nifti1Image(synthetic_us_image(rng).T[:, :, None].astype(np.float32), np.eye(4)),
             str(inp / "nifti_case.nii.gz"))
    dcm = sitk.GetImageFromArray(synthetic_us_image(rng))
    dcm.SetSpacing((0.1, 0.1))
    sitk.WriteImage(dcm, str(inp / "dicom_case" / "image.dcm"))
    n_cases = 5

    models = {
        "incl_excl": (REPO / "models/classifiers/incl_excl/cp.keras", ["exclude", "include"], "include"),
        "cor_sag": (REPO / "models/classifiers/cor_sag/cp.keras", ["Coronal", "Saggital"], "Saggital"),
    }
    for name, (model_path, class_names, positive) in models.items():
        if not model_path.exists():
            report(stage, f"{name}: model file", "FAIL", f"{model_path} not found")
            continue
        out_csv = work / f"cls_{name}.csv"
        ok, secs = run([sys.executable, SRC / "classification/classify.py",
                        "--input", inp, "--tf-model", model_path,
                        "--harmonized-dir", work / "harmonized",
                        "--output-csv", out_csv,
                        "--class-names", *class_names, "--positive-class", positive])
        if not ok or not out_csv.exists():
            report(stage, f"{name}: classify.py runs", "FAIL")
            continue
        report(stage, f"{name}: classify.py runs", "PASS", f"{secs:.0f}s")

        df = pd.read_csv(out_csv)
        expected_cols = {"case_id", "kind", "status", "nifti_path", "classifier_label",
                         "classifier_confidence", "included"}
        problems = []
        if not expected_cols.issubset(df.columns):
            problems.append(f"missing columns {expected_cols - set(df.columns)}")
        if len(df) != n_cases:
            problems.append(f"{len(df)} rows, expected {n_cases}")
        if (df["status"] != "classified").any():
            problems.append(f"status: {df['status'].value_counts().to_dict()}")
        if not df["classifier_label"].dropna().isin(class_names).all():
            problems.append("unknown labels")
        conf = df["classifier_confidence"].dropna()
        if not ((conf >= 0) & (conf <= 1)).all():
            problems.append("confidence outside [0, 1]")
        if not all(Path(p).exists() for p in df["nifti_path"].dropna()):
            problems.append("harmonized NIfTI missing")
        kinds = sorted(df["kind"].unique())
        report(stage, f"{name}: output CSV (png, nifti, dicom)", "FAIL" if problems else "PASS",
               "; ".join(problems) or f"kinds={kinds}, labels={df['classifier_label'].value_counts().to_dict()}")


# ---------------------------------------------------------------------------
# Stage: segmentation
# ---------------------------------------------------------------------------

def stage_segmentation(work: Path, rng: np.random.Generator, nnunet_model, device: str) -> None:
    stage = "segmentation"
    missing = [m for m in ("nibabel", "scipy") if not has_module(m)]
    if missing:
        report(stage, "dependencies", "SKIP", f"missing: {', '.join(missing)}")
        return

    import nibabel as nib

    # make_vermis_labelmap.py on synthetic nnU-Net-style predictions
    cer, sku, out = work / "pred_cerebellum", work / "pred_skull", work / "labelmaps"
    cer.mkdir()
    sku.mkdir()
    affine = np.diag([-0.1, -0.1, 1.0, 1.0])
    c = np.zeros((120, 100, 1), np.uint8)
    c[40:80, 30:60] = 1
    c[5:8, 5:8] = 1                            # small false-positive island
    s = np.zeros_like(c)
    s[:, :6] = 1
    nib.save(nib.Nifti1Image(c, affine), str(cer / "case_a.nii.gz"))
    nib.save(nib.Nifti1Image(s, affine), str(sku / "case_a.nii.gz"))
    nib.save(nib.Nifti1Image(np.zeros_like(c), affine), str(cer / "case_empty.nii.gz"))

    ok, _ = run([sys.executable, SRC / "segmentation/make_vermis_labelmap.py",
                 "--cerebellum-dir", cer, "--skull-dir", sku, "--output-dir", out, "--largest-component"])
    if not ok:
        report(stage, "make_vermis_labelmap.py runs", "FAIL")
    else:
        res = nib.load(str(out / "case_a.nii.gz"))
        v = np.asanyarray(res.dataobj)
        problems = []
        if (v == 2).sum() != 40 * 30:
            problems.append(f"cerebellum pixels {(v == 2).sum()} != {40 * 30} (island not removed?)")
        if (v == 1).sum() == 0:
            problems.append("skull label missing")
        if not np.allclose(res.affine, affine):
            problems.append("affine changed")
        if (out / "case_empty.nii.gz").exists():
            problems.append("empty prediction was not skipped")
        report(stage, "make_vermis_labelmap.py output", "FAIL" if problems else "PASS", "; ".join(problems))

    # segment.py with a real trained model
    if nnunet_model is None:
        report(stage, "segment.py with nnU-Net", "SKIP", "pass --nnunet-model to test")
        return
    if not has_module("nnunetv2"):
        report(stage, "segment.py with nnU-Net", "SKIP", "nnunetv2 not installed")
        return

    import pandas as pd
    harm = work / "seg_input"
    harm.mkdir()
    rows = []
    for i in range(2):
        p = harm / f"seg_case_{i}.nii.gz"
        img = synthetic_us_image(rng).T[:, :, None].astype(np.float32)
        nib.save(nib.Nifti1Image(img, np.diag([0.1, 0.1, 1.0, 1.0])), str(p))
        rows.append({"case_id": p.name[:-7], "nifti_path": str(p), "original_path": None, "included": True})
    csv = work / "seg_cases.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)

    manifest = work / "seg_manifest.csv"
    ok, secs = run([sys.executable, SRC / "segmentation/segment.py",
                    "--classifier-csv", csv, "--output-dir", work / "seg_out",
                    "--nnunet-model-folder", nnunet_model, "--device", device,
                    "--manifest-out", manifest])
    if not ok or not manifest.exists():
        report(stage, "segment.py with nnU-Net", "FAIL")
        return
    m = pd.read_csv(manifest)
    n_ok = int((m["status"] == "segmented").sum())
    report(stage, "segment.py with nnU-Net", "PASS" if n_ok == len(rows) else "FAIL",
           f"{n_ok}/{len(rows)} segmented in {secs:.0f}s")


# ---------------------------------------------------------------------------
# Stage: vermis height
# ---------------------------------------------------------------------------

def stage_vermis(work: Path, rng: np.random.Generator) -> None:
    stage = "vermis"
    missing = [m for m in ("pycpd", "cv2", "nibabel", "pandas", "scipy", "matplotlib", "pydicom", "pydantic")
               if not has_module(m)]
    if missing:
        report(stage, "dependencies", "SKIP", f"missing: {', '.join(missing)}")
        return

    import nibabel as nib
    import pandas as pd

    data = work / "ssm_data"
    truth = make_vermis_cases(data, rng)
    report(stage, "synthetic data (01 layout)", "PASS", f"{len(truth)} cases")

    vh = SRC / "vermis_height"
    steps = [
        ("02 correspondence", [vh / "02_correspondent_landmarking.py", "--data-dir", data,
                               "--output-dir", data / "correspondence", "--n-landmarks", 100,
                               "--target-density", 400, "--cpd-alpha", 4.0]),
        ("03 PCA", [vh / "03_build_pca.py", "--correspondence-dir", data / "correspondence",
                    "--output-dir", data / "pca"]),
        ("04 fit", [vh / "04_fit_pca.py", "--pca-model", data / "pca/sagittal/pca_shape_model.npz",
                    "--input", data / "labels/sagittal", "--output-dir", data / "fitted/sagittal",
                    "--max-hd95-mm", MAX_HD95_MM]),
    ]
    for name, cmd in steps:
        ok, secs = run([sys.executable, *cmd])
        report(stage, name, "PASS" if ok else "FAIL", f"{secs:.0f}s")
        if not ok:
            return

    n_corr = len(list((data / "correspondence/sagittal/landmarks").glob("*.csv"))) \
        if (data / "correspondence/sagittal/landmarks").exists() else None
    if n_corr is not None:
        report(stage, "02 kept samples", "INFO",
               f"{n_corr}/{N_VERMIS_SAMPLES} pass the CPD residual filter and go into the PCA")

    fit = pd.read_csv(data / "fitted/sagittal/fit_results.csv")
    ok_fits = fit[fit["success"] == True]  # noqa: E712
    hd95 = ok_fits["hd95_mm"]
    n_good = int(ok_fits["fit_quality_ok"].sum()) if "fit_quality_ok" in ok_fits else 0
    report(stage, "04 fit quality (HD95)",
           "PASS" if len(ok_fits) == N_VERMIS_SAMPLES and n_good >= MIN_MEASURED_FRACTION * N_VERMIS_SAMPLES
           else "FAIL",
           f"{len(ok_fits)}/{N_VERMIS_SAMPLES} fitted, {n_good} with HD95 <= {MAX_HD95_MM} mm, "
           f"HD95 median {hd95.median():.3f} mm, max {hd95.max():.3f} mm")

    # Which landmark indices are the top/bottom of the synthetic vermis?
    # Landmark coordinates live in the NIfTI array space of 01 (array is
    # stored transposed: (W, H)). Find the array axis along which the
    # vermis is longest = image vertical, and take the extreme indices.
    lab = np.squeeze(nib.load(str(next((data / "labels/sagittal").glob("*.nii.gz")))).get_fdata())
    ys, xs = np.nonzero(lab == 2)
    long_axis_is_x = (xs.max() - xs.min()) > (ys.max() - ys.min())   # cv2 x = array axis 1
    coord = "x" if long_axis_is_x else "y"
    lm_files = sorted((data / "fitted/sagittal/fitted_landmarks").glob("*.csv"))
    mean_coord = np.mean([pd.read_csv(f)[coord].to_numpy() for f in lm_files], axis=0)
    lm_a, lm_b = int(np.argmin(mean_coord)), int(np.argmax(mean_coord))
    report(stage, "landmark indices", "INFO",
           f"top/bottom of synthetic vermis = landmarks {lm_a} and {lm_b}; vermis long axis is landmark-'{coord}'")

    # 06 with the production angle setting, with and without adaptive anchors
    for variant, extra in [("", []), (" (adaptive window 2)", ["--adaptive-search-window", 2])]:
        meas = data / ("measurements/sagittal" + ("_adaptive" if extra else ""))
        ok, secs = run([sys.executable, vh / "06_measure_vermis.py",
                        "--fitted-landmarks-dir", data / "fitted/sagittal/fitted_landmarks",
                        "--fit-results-csv", data / "fitted/sagittal/fit_results.csv",
                        "--spacing-dir", data / "images/sagittal",
                        "--images-dir", data / "images/sagittal",
                        "--labels-dir", data / "labels/sagittal",
                        "--output-dir", meas,
                        "--landmark-a", lm_a, "--landmark-b", lm_b,
                        "--max-angle-deviation-deg", 25, *extra])
        report(stage, "06 measure" + variant, "PASS" if ok else "FAIL", f"{secs:.0f}s")
        if not ok:
            return
        check_measurements(stage, variant, pd.read_csv(meas / "measurements_summary.csv"), truth)


def check_measurements(stage: str, variant: str, summ, truth: dict) -> None:
    n_reliable = int(summ["measurement_reliable"].sum()) if "measurement_reliable" in summ else 0
    report(stage, "06 verticality check" + variant, "PASS" if n_reliable == len(summ) else "FAIL",
           f"{n_reliable}/{len(summ)} marked measurement_reliable "
           "(the synthetic vermis is vertical in the original image)")

    summ["truth_mm"] = summ["sample_id"].map(truth)
    valid = summ.dropna(subset=["trimmed_mean_mm", "truth_mm"])
    if valid.empty:
        report(stage, "06 height vs. known height" + variant, "FAIL", "no usable measurements")
        return
    rel_err = (valid["trimmed_mean_mm"] - valid["truth_mm"]).abs() / valid["truth_mm"]
    mae = (valid["trimmed_mean_mm"] - valid["truth_mm"]).abs().mean()
    report(stage, "06 height vs. known height" + variant,
           "PASS" if rel_err.max() <= HEIGHT_TOLERANCE and len(valid) >= MIN_MEASURED_FRACTION * N_VERMIS_SAMPLES
           else "FAIL",
           f"n={len(valid)}, MAE {mae:.2f} mm, max rel. error {rel_err.max():.1%} "
           f"(truth {valid['truth_mm'].min():.1f}-{valid['truth_mm'].max():.1f} mm)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", nargs="+", default=["classification", "segmentation", "vermis"],
                    choices=["classification", "segmentation", "vermis"])
    ap.add_argument("--nnunet-model", type=Path, default=None,
                    help="Trained nnU-Net model folder (trainer__plans__config) to test segment.py with")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    ap.add_argument("--workdir", type=Path, default=None, help="Keep outputs here instead of a temp dir")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    work = args.workdir or Path(tempfile.mkdtemp(prefix="neus_smoke_"))
    if args.workdir and work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    print(f"Working directory: {work}\n")
    rng = np.random.default_rng(args.seed)

    if "classification" in args.stages:
        stage_classification(work, rng)
    if "segmentation" in args.stages:
        stage_segmentation(work, rng, args.nnunet_model, args.device)
    if "vermis" in args.stages:
        stage_vermis(work, rng)

    n_fail = sum(r[2] == "FAIL" for r in RESULTS)
    n_pass = sum(r[2] == "PASS" for r in RESULTS)
    n_skip = sum(r[2] == "SKIP" for r in RESULTS)
    print(f"\n{n_pass} passed, {n_fail} failed, {n_skip} skipped")
    if args.workdir is None and n_fail == 0:
        shutil.rmtree(work, ignore_errors=True)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
