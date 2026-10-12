"""
Combine nnU-Net predictions into the labelmap convention used by the
vermis height pipeline (src/vermis_height/04_fit_pca.py):

    0 = background
    1 = skull
    2 = cerebellum / vermis

The cerebellum model (Dataset010_midsagittal_cerebellum) predicts a binary
mask (1 = cerebellum). The skull model is optional - the shape model only
uses label 2, the skull is kept for visualization.

The NIfTI header (spacing/affine) of the cerebellum prediction is kept, so
the output can also be passed as --spacing-dir to 06_measure_vermis.py.

Usage:
    python make_vermis_labelmap.py \\
        --cerebellum-dir /path/to/cerebellum_predictions \\
        [--skull-dir /path/to/skull_predictions] \\
        [--skull-label 1] \\
        --output-dir /path/to/vermis_labelmaps \\
        [--largest-component]

Requirements
------------
    pip install nibabel numpy scipy
"""

import argparse
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
from scipy import ndimage


SKULL_VALUE = 1
CEREBELLUM_VALUE = 2


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    labeled, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    return labeled == (int(np.argmax(sizes)) + 1)


def find_matching(directory: Optional[Path], name: str) -> Optional[Path]:
    if directory is None:
        return None
    candidate = directory / name
    return candidate if candidate.exists() else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cerebellum-dir", type=Path, required=True,
                    help="nnU-Net output dir of the cerebellum model (foreground = 1)")
    ap.add_argument("--skull-dir", type=Path, default=None,
                    help="Optional nnU-Net output dir of the skull model, same file names")
    ap.add_argument("--skull-label", type=int, default=1,
                    help="Label value of the skull in the skull model output (default 1)")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--largest-component", action="store_true",
                    help="Keep only the largest connected cerebellum component")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(args.cerebellum_dir.glob("*.nii.gz")) + sorted(args.cerebellum_dir.glob("*.nii"))
    if not files:
        raise SystemExit(f"No NIfTI predictions found in {args.cerebellum_dir}")

    n_written, n_empty = 0, 0
    for cerebellum_path in files:
        nii = nib.load(str(cerebellum_path))
        cerebellum = np.asanyarray(nii.dataobj) > 0

        if args.largest_component:
            cerebellum = keep_largest_component(cerebellum)

        if not cerebellum.any():
            print(f"    {cerebellum_path.name}: empty cerebellum prediction - skipping")
            n_empty += 1
            continue

        out = np.zeros(cerebellum.shape, dtype=np.int16)

        skull_path = find_matching(args.skull_dir, cerebellum_path.name)
        if skull_path is not None:
            skull = np.asanyarray(nib.load(str(skull_path)).dataobj)
            if skull.shape == out.shape:
                out[skull == args.skull_label] = SKULL_VALUE
            else:
                print(f"    {cerebellum_path.name}: skull shape {skull.shape} != {out.shape} - skull ignored")

        out[cerebellum] = CEREBELLUM_VALUE

        out_nii = nib.Nifti1Image(out, nii.affine, nii.header)
        out_nii.set_data_dtype(np.int16)
        nib.save(out_nii, str(args.output_dir / cerebellum_path.name))
        n_written += 1

    print(f"\n{n_written} labelmap(s) written to {args.output_dir} ({n_empty} empty prediction(s) skipped)")


if __name__ == "__main__":
    main()
