"""
Track vermis height development over time, per patient - plotted
against corrected (post-menstrual) age, accounting for gestational
age at birth, since this is a preterm ("preemie") cohort where
postnatal day-of-life alone is not a meaningful developmental axis
across patients born at different gestational ages.

Joins the per-image measurements (output of 06_measure_vermis.py) with
each image's acquisition date/time, PatientAge, and PatientID (read
from the original DICOM via matched_pairs.csv, output of
01_preprocess_data.py), then:

    1. Computes an exact POSTNATAL age-in-days for every scan.
       PatientAge in DICOM is often coarse (e.g. "006W" = 6 weeks,
       only +/-7 day precision) and later scans sometimes only have
       week/month precision even when an earlier scan had day
       precision. To get an exact age for every scan of a patient:
       take the earliest scan that has ANY usable PatientAge tag, use
       it to derive an approximate birth date (acquisition date -
       parsed age), then compute every scan's exact age as (that
       scan's acquisition date - approximate birth date).

    2. Parses gestational age AT BIRTH from PatientID, which this
       institution encodes like "NB-Pat.004-SSW28+1-LT1" (German
       "Schwangerschaftswoche" = week of pregnancy: 28 weeks + 1 day
       gestational age at birth).

    3. Computes CORRECTED (post-menstrual) age for every scan:
           corrected_age_days = postnatal age_days + gestational age
                                 at birth (days)
       This is the standard neonatology way to compare preterm
       infants born at different gestational ages on the same
       developmental timeline, and is the more meaningful axis for a
       developmental measurement like vermis height. Both age_days
       (postnatal) and corrected_age_days are kept; --age-column
       selects which one plots use (default: corrected_age_days).

    4. Writes one CSV per patient, all their measurements (sagittal
       and/or coronal) ordered chronologically.
    5. Writes a combined all-patients CSV.
    6. Writes a per-patient summary (n scans, age range, first/last
       height, change).
    7. Plots, all scatter (no boxplots for timelines) - each visit's
       5 raw measurements (from 06_measure_vermis.py) are only there
       to produce ONE stabilized value per visit; only that one value
       is plotted, not all 5:
         - one scatter+trend-line plot per patient (height vs. age,
           per plane)
         - a cohort-level scatter per plane (every patient's value,
           no connecting lines - stays readable at any cohort size)
         - a cohort-level spaghetti plot per plane (one line per
           patient - for directly comparing patients' trajectories)
         - a cohort-level growth-chart per plane (median trend line +
           25th-75th percentile band, binned by age - "what's typical
           at this age" and where an individual patient stands out)
         - a summary boxplot comparing height distributions between
           planes across the whole cohort

Date/time is taken from, in order of preference: AcquisitionDate/Time,
ContentDate/Time, SeriesDate/Time, StudyDate/Time - whichever is
populated first in that DICOM.
"""

import argparse
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict

import numpy as np
import pandas as pd
import pydicom
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CLASSES = ("sagittal", "coronal")

DATE_TIME_TAG_PAIRS = (
    ("AcquisitionDate", "AcquisitionTime"),
    ("ContentDate", "ContentTime"),
    ("SeriesDate", "SeriesTime"),
    ("StudyDate", "StudyTime"),
)

# DICOM Age String (AS) units -> days. Months/years use calendar
# averages since AS doesn't carry enough precision to do better.
AGE_UNIT_TO_DAYS = {"D": 1.0, "W": 7.0, "M": 30.4368, "Y": 365.2425}
AGE_PATTERN = re.compile(r"^\s*0*(\d+)\s*([DWMYdwmy])\s*$")

# Gestational age at birth, encoded in PatientID like
# "NB-Pat.004-SSW28+1-LT1" -> 28 weeks + 1 day (SSW = German
# "Schwangerschaftswoche", week of pregnancy). The "+N" day part is
# optional (defaults to 0 if absent).
GESTATIONAL_AGE_PATTERN = re.compile(r"SSW\s*0*(\d+)(?:\s*\+\s*0*(\d+))?", re.IGNORECASE)

VALUE_COLUMN_DEFAULT = "trimmed_mean_mm"
AGE_COLUMN_DEFAULT = "corrected_age_days"
STAT_COLUMNS = ["mean_mm", "median_mm", "trimmed_mean_mm", "std_mm", "min_mm", "max_mm"]

AGE_COLUMN_LABELS = {
    "age_days": "Postnatal age (days since birth)",
    "corrected_age_days": "Corrected (post-menstrual) age (days)",
}


# ============================================================
# DICOM metadata extraction: acquisition date/time, PatientAge,
# gestational age at birth (from PatientID)
# ============================================================

def parse_patient_age(age_str) -> Optional[float]:
    """
    Parse a DICOM PatientAge (VR: AS) string like "006W", "5D", "2Y"
    into an approximate age in days. Returns None if unparseable or
    empty.
    """

    if not age_str:
        return None

    match = AGE_PATTERN.match(str(age_str))

    if not match:
        return None

    value = int(match.group(1))
    unit = match.group(2).upper()

    return value * AGE_UNIT_TO_DAYS[unit]


def parse_gestational_age_at_birth(patient_id) -> Optional[float]:
    """
    Parse gestational age at birth from a PatientID like
    "NB-Pat.004-SSW28+1-LT1" -> 28*7 + 1 = 197 days. Returns None if
    no SSW pattern is found.
    """

    if not patient_id:
        return None

    match = GESTATIONAL_AGE_PATTERN.search(str(patient_id))

    if not match:
        return None

    weeks = int(match.group(1))
    extra_days = int(match.group(2)) if match.group(2) else 0

    return weeks * 7 + extra_days


def read_dicom_metadata(dcm_path: str) -> dict:

    result = {
        "acquisition_datetime": None,
        "patient_age_raw": None,
        "patient_age_days_tag": None,
        "gestational_age_at_birth_days_raw": None,
    }

    try:
        ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
    except Exception as e:
        print(f"    Cannot read DICOM {dcm_path}: {e}")
        return result

    for date_attr, time_attr in DATE_TIME_TAG_PAIRS:

        date_val = getattr(ds, date_attr, None)

        if not date_val:
            continue

        time_val = getattr(ds, time_attr, None)

        date_str = str(date_val)[:8]
        time_str = str(time_val)[:6].ljust(6, "0") if time_val else "000000"

        try:
            result["acquisition_datetime"] = datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")
            break
        except ValueError:
            continue

    age_raw = getattr(ds, "PatientAge", None)

    if age_raw:
        result["patient_age_raw"] = str(age_raw)
        result["patient_age_days_tag"] = parse_patient_age(age_raw)

    patient_id_raw = getattr(ds, "PatientID", None)

    if patient_id_raw:
        result["gestational_age_at_birth_days_raw"] = parse_gestational_age_at_birth(patient_id_raw)

    return result


# ============================================================
# Load + join measurements with patient/date/age metadata
# ============================================================

def load_all_measurements(
    sagittal_measurements: Optional[Path],
    coronal_measurements: Optional[Path],
) -> pd.DataFrame:

    frames = []

    for class_name, path in (("sagittal", sagittal_measurements), ("coronal", coronal_measurements)):

        if path is None:
            continue

        if not path.exists():
            print(f"    Warning: {path} not found; skipping {class_name}")
            continue

        df = pd.read_csv(path)
        df["class"] = class_name
        frames.append(df)

    if not frames:
        raise ValueError("No measurement files found (need at least one of sagittal/coronal)")

    return pd.concat(frames, ignore_index=True)


def build_dated_table(
    measurements_df: pd.DataFrame,
    matched_pairs_df: pd.DataFrame,
) -> pd.DataFrame:

    required_cols = {"image_id", "patient_id", "image_path", "class"}
    missing = required_cols - set(matched_pairs_df.columns)

    if missing:
        raise ValueError(f"matched_pairs.csv is missing columns: {missing}")

    merged = measurements_df.merge(
        matched_pairs_df[["image_id", "patient_id", "image_path", "class"]],
        left_on=["sample_id", "class"],
        right_on=["image_id", "class"],
        how="left",
    )

    n_unmatched = merged["patient_id"].isna().sum()
    if n_unmatched > 0:
        print(
            f"    Warning: {n_unmatched} measurement rows could not be matched "
            f"to matched_pairs.csv (sample_id/class not found) and will be dropped"
        )
    merged = merged.dropna(subset=["patient_id", "image_path"]).copy()

    print("    Reading acquisition date/time, PatientAge, and PatientID from DICOMs...")

    meta_cache: Dict[str, dict] = {}

    def get_meta(path: str) -> dict:
        if path not in meta_cache:
            meta_cache[path] = read_dicom_metadata(path)
        return meta_cache[path]

    metas = merged["image_path"].apply(get_meta)
    merged["acquisition_datetime"] = metas.apply(lambda m: m["acquisition_datetime"])
    merged["patient_age_raw"] = metas.apply(lambda m: m["patient_age_raw"])
    merged["patient_age_days_tag"] = metas.apply(lambda m: m["patient_age_days_tag"])
    merged["gestational_age_at_birth_days_raw"] = metas.apply(lambda m: m["gestational_age_at_birth_days_raw"])

    n_no_date = merged["acquisition_datetime"].isna().sum()
    if n_no_date > 0:
        print(f"    Warning: {n_no_date} rows have no usable date/time and will be dropped")
    merged = merged.dropna(subset=["acquisition_datetime"]).copy()

    merged = merged.sort_values(["patient_id", "acquisition_datetime"])

    return merged


def compute_exact_age_days(dated_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each patient: use the earliest scan that has ANY usable
    PatientAge tag as a reference to derive an approximate birth
    date, then compute every scan's exact POSTNATAL age (in days)
    from that reference using the acquisition date's exact
    precision. Patients with no usable PatientAge tag on any scan
    get age_days = NaN.
    """

    dated_df = dated_df.copy()
    dated_df["age_days"] = np.nan

    n_patients_no_age = 0

    for patient_id, group in dated_df.groupby("patient_id"):

        group_sorted = group.sort_values("acquisition_datetime")

        reference_idx = None
        for idx, row in group_sorted.iterrows():
            if pd.notna(row["patient_age_days_tag"]):
                reference_idx = idx
                break

        if reference_idx is None:
            n_patients_no_age += 1
            continue

        reference_row = dated_df.loc[reference_idx]
        birth_date_approx = reference_row["acquisition_datetime"] - pd.to_timedelta(
            reference_row["patient_age_days_tag"], unit="D"
        )

        idxs = group.index
        dated_df.loc[idxs, "age_days"] = (
            (dated_df.loc[idxs, "acquisition_datetime"] - birth_date_approx).dt.total_seconds() / 86400.0
        )

    if n_patients_no_age > 0:
        print(
            f"    Warning: {n_patients_no_age} patient(s) have no usable PatientAge tag on "
            f"any scan; their age_days is NaN and they will be excluded from age-based plots"
        )

    return dated_df


def resolve_gestational_age_and_correct(dated_df: pd.DataFrame) -> pd.DataFrame:
    """
    Resolve one gestational-age-at-birth value per patient (from
    PatientID's SSW.. pattern - should be constant per patient, but
    resolved defensively via majority vote across their scans in
    case of formatting drift), then compute corrected_age_days =
    age_days (postnatal) + gestational_age_at_birth_days.
    """

    dated_df = dated_df.copy()

    resolved: Dict[str, float] = {}
    n_no_ga = 0
    n_inconsistent = 0

    for patient_id, group in dated_df.groupby("patient_id"):

        values = group["gestational_age_at_birth_days_raw"].dropna()

        if len(values) == 0:
            resolved[patient_id] = np.nan
            n_no_ga += 1
            continue

        unique_values = values.unique()

        if len(unique_values) > 1:
            n_inconsistent += 1
            chosen = values.value_counts().idxmax()
            print(
                f"    Warning: patient {patient_id} has inconsistent gestational age "
                f"values across scans {sorted(unique_values)} days - using the most common: {chosen} days"
            )
            resolved[patient_id] = chosen
        else:
            resolved[patient_id] = unique_values[0]

    dated_df["gestational_age_at_birth_days"] = dated_df["patient_id"].map(resolved)
    dated_df["gestational_age_at_birth_weeks"] = dated_df["gestational_age_at_birth_days"] / 7.0

    dated_df["corrected_age_days"] = dated_df["age_days"] + dated_df["gestational_age_at_birth_days"]
    dated_df["corrected_age_weeks"] = dated_df["corrected_age_days"] / 7.0

    if n_no_ga > 0:
        print(
            f"    Warning: {n_no_ga} patient(s) have no parseable gestational age "
            f"(no 'SSW..' pattern in PatientID); corrected_age_days is NaN for them - "
            f"they will be excluded from corrected-age-based plots (use --age-column age_days "
            f"to include them via postnatal age instead)"
        )
    if n_inconsistent > 0:
        print(f"    Warning: {n_inconsistent} patient(s) had inconsistent gestational age strings across their scans")

    return dated_df


# ============================================================
# Per-patient CSVs + summary
# ============================================================

def write_per_patient_csvs(
    dated_df: pd.DataFrame,
    output_dir: Path,
    value_column: str,
    age_column: str,
) -> pd.DataFrame:

    per_patient_dir = output_dir / "per_patient"
    per_patient_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    output_cols = [
        "acquisition_datetime", "age_days", "patient_age_raw",
        "gestational_age_at_birth_weeks", "corrected_age_days", "corrected_age_weeks",
        "class", "sample_id",
    ] + STAT_COLUMNS

    for patient_id, group in dated_df.groupby("patient_id"):

        group = group.sort_values("acquisition_datetime").copy()

        group[output_cols].to_csv(per_patient_dir / f"{patient_id}.csv", index=False)

        for class_name, class_group in group.groupby("class"):

            class_group = class_group.dropna(subset=[age_column]).sort_values(age_column)

            if len(class_group) == 0:
                continue

            summary_rows.append(
                {
                    "patient_id": patient_id,
                    "class": class_name,
                    "n_scans": len(class_group),
                    "gestational_age_at_birth_weeks": class_group["gestational_age_at_birth_weeks"].iloc[0],
                    f"first_{age_column}": class_group[age_column].iloc[0],
                    f"last_{age_column}": class_group[age_column].iloc[-1],
                    f"first_{value_column}": class_group[value_column].iloc[0],
                    f"last_{value_column}": class_group[value_column].iloc[-1],
                    "change": class_group[value_column].iloc[-1] - class_group[value_column].iloc[0],
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "per_patient_summary.csv", index=False)

    return summary_df


# ============================================================
# Plots
# ============================================================

def plot_patient_timeline(
    patient_id: str,
    group: pd.DataFrame,
    value_column: str,
    age_column: str,
    output_path: Path,
):

    group = group.dropna(subset=[age_column])

    if len(group) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {"sagittal": "tab:blue", "coronal": "tab:orange"}

    for class_name, class_group in group.groupby("class"):

        class_group = class_group.sort_values(age_column)
        color = colors.get(class_name, "gray")

        ax.scatter(
            class_group[age_column], class_group[value_column],
            color=color, label=class_name, s=50, zorder=3,
        )

        # Trend line connecting this patient's points for this plane,
        # in age order (a no-op visually if there's only one point).
        ax.plot(
            class_group[age_column], class_group[value_column],
            color=color, linewidth=1.5, alpha=0.7, zorder=2,
        )

    ax.set_xlabel(AGE_COLUMN_LABELS.get(age_column, age_column))
    ax.set_ylabel(f"Vermis height ({value_column}, mm)")
    ax.set_title(f"Patient {patient_id}: vermis height vs. age")
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_cohort_overview(
    dated_df: pd.DataFrame,
    class_name: str,
    value_column: str,
    age_column: str,
    output_path: Path,
):
    """
    A single scatter of every patient's stabilized value against
    age - deliberately not per-patient lines, which turn into an
    unreadable mess once there are more than a handful of patients.
    """

    class_df = dated_df[dated_df["class"] == class_name].dropna(subset=[age_column])

    if len(class_df) == 0:
        return

    fig, ax = plt.subplots(figsize=(9, 6))

    color = "tab:blue" if class_name == "sagittal" else "tab:orange"

    ax.scatter(class_df[age_column], class_df[value_column], alpha=0.6, s=30, color=color)

    ax.set_xlabel(AGE_COLUMN_LABELS.get(age_column, age_column))
    ax.set_ylabel(f"Vermis height ({value_column}, mm)")
    ax.set_title(f"{class_name.capitalize()}: vermis height vs. age (all patients)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_cohort_spaghetti(
    dated_df: pd.DataFrame,
    class_name: str,
    value_column: str,
    age_column: str,
    output_path: Path,
):
    """
    One line per patient, all overlaid - useful specifically for
    comparing patients' trajectories against each other (is this
    patient growing faster/slower than others, starting higher/
    lower, etc.), which a plain scatter can't show.
    """

    class_df = dated_df[dated_df["class"] == class_name].dropna(subset=[age_column])

    if len(class_df) == 0:
        return

    patient_ids = sorted(class_df["patient_id"].unique())

    if len(patient_ids) == 0:
        return

    fig, ax = plt.subplots(figsize=(9, 6))

    colors = plt.cm.viridis(np.linspace(0, 1, len(patient_ids)))

    for color, patient_id in zip(colors, patient_ids):

        patient_group = class_df[class_df["patient_id"] == patient_id].sort_values(age_column)

        ax.plot(
            patient_group[age_column], patient_group[value_column],
            marker="o", markersize=4, linewidth=1.2, alpha=0.6, color=color,
        )

    ax.set_xlabel(AGE_COLUMN_LABELS.get(age_column, age_column))
    ax.set_ylabel(f"Vermis height ({value_column}, mm)")
    ax.set_title(f"{class_name.capitalize()}: per-patient trajectories vs. age")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_cohort_growth_chart(
    dated_df: pd.DataFrame,
    class_name: str,
    value_column: str,
    age_column: str,
    age_bin_days: float,
    output_path: Path,
):
    """
    A pediatric-growth-chart-style view: bins all patients' values by
    age, then shows the median trend line with a 25th-75th percentile
    shaded band, with the raw points faint underneath for context.
    """

    class_df = dated_df[dated_df["class"] == class_name].dropna(subset=[age_column])

    if len(class_df) == 0:
        return

    max_age = class_df[age_column].max()
    n_bins = max(1, int(np.ceil((max_age + 1e-6) / age_bin_days)))
    bin_edges = np.arange(n_bins + 1) * age_bin_days

    class_df = class_df.copy()
    class_df["age_bin"] = pd.cut(class_df[age_column], bins=bin_edges, right=False, include_lowest=True)

    bin_centers = []
    medians = []
    p25s = []
    p75s = []

    for interval in sorted(class_df["age_bin"].dropna().unique()):

        bin_values = class_df.loc[class_df["age_bin"] == interval, value_column].dropna()

        if len(bin_values) == 0:
            continue

        bin_centers.append((interval.left + interval.right) / 2.0)
        medians.append(bin_values.median())
        p25s.append(bin_values.quantile(0.25))
        p75s.append(bin_values.quantile(0.75))

    if not bin_centers:
        return

    color = "tab:blue" if class_name == "sagittal" else "tab:orange"

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(class_df[age_column], class_df[value_column], alpha=0.15, s=20, color="gray")
    ax.fill_between(bin_centers, p25s, p75s, color=color, alpha=0.25, label="25th-75th percentile")
    ax.plot(bin_centers, medians, marker="o", color=color, linewidth=2, label="median")

    ax.set_xlabel(AGE_COLUMN_LABELS.get(age_column, age_column))
    ax.set_ylabel(f"Vermis height ({value_column}, mm)")
    ax.set_title(f"{class_name.capitalize()}: vermis height growth chart (bin width = {age_bin_days:.0f} days)")
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_summary_boxplot(
    dated_df: pd.DataFrame,
    value_column: str,
    output_path: Path,
):

    data = [
        dated_df[dated_df["class"] == class_name][value_column].dropna().to_numpy()
        for class_name in CLASSES
        if (dated_df["class"] == class_name).any()
    ]
    labels = [c for c in CLASSES if (dated_df["class"] == c).any()]

    if not data:
        return

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.boxplot(data, tick_labels=labels, showmeans=True)
    ax.set_ylabel(f"Vermis height ({value_column}, mm)")
    ax.set_title("Vermis height distribution across all scans")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--matched-pairs-csv", required=True, help="matched_pairs.csv from 01_preprocess_data.py")
    ap.add_argument("--sagittal-measurements", default=None, help="measurements_summary.csv for sagittal (06_measure_vermis.py output)")
    ap.add_argument("--coronal-measurements", default=None, help="measurements_summary.csv for coronal (06_measure_vermis.py output)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument(
        "--value-column", default=VALUE_COLUMN_DEFAULT,
        choices=["mean_mm", "median_mm", "trimmed_mean_mm"],
        help="Which single stabilized statistic to plot/report (default: trimmed_mean_mm). The other 4 offset measurements only inform this value; they are not plotted separately.",
    )
    ap.add_argument(
        "--age-column", default=AGE_COLUMN_DEFAULT,
        choices=["age_days", "corrected_age_days"],
        help=(
            "Which age to plot against (default: corrected_age_days, i.e. post-menstrual "
            "age = postnatal age + gestational age at birth - the standard way to compare "
            "preterm infants developmentally). Use age_days for plain postnatal age since birth."
        ),
    )
    ap.add_argument(
        "--age-bin-days", type=float, default=30.0,
        help="Age bin width (days) for the growth-chart plot's median/percentile bands (default 30)",
    )

    args = ap.parse_args()

    if not args.sagittal_measurements and not args.coronal_measurements:
        raise ValueError("Provide at least one of --sagittal-measurements / --coronal-measurements")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    matched_pairs_df = pd.read_csv(args.matched_pairs_csv)

    measurements_df = load_all_measurements(
        Path(args.sagittal_measurements) if args.sagittal_measurements else None,
        Path(args.coronal_measurements) if args.coronal_measurements else None,
    )

    dated_df = build_dated_table(measurements_df, matched_pairs_df)

    if len(dated_df) == 0:
        print("No measurements could be dated/matched to a patient; nothing to do.")
        return

    print(f"Computing exact postnatal age (days) for {dated_df['patient_id'].nunique()} patients...")
    dated_df = compute_exact_age_days(dated_df)

    print("Resolving gestational age at birth and computing corrected age...")
    dated_df = resolve_gestational_age_and_correct(dated_df)

    dated_df.to_csv(output_dir / "all_measurements_dated.csv", index=False)

    print(f"\nWriting per-patient CSVs for {dated_df['patient_id'].nunique()} patients...")
    write_per_patient_csvs(dated_df, output_dir, args.value_column, args.age_column)

    print(f"Generating plots (age axis: {args.age_column})...")

    plots_dir = output_dir / "plots"

    for patient_id, group in dated_df.groupby("patient_id"):
        plot_patient_timeline(
            patient_id, group, args.value_column, args.age_column,
            plots_dir / "per_patient" / f"{patient_id}.png",
        )

    for class_name in CLASSES:
        plot_cohort_overview(
            dated_df, class_name, args.value_column, args.age_column,
            plots_dir / f"cohort_overview_{class_name}.png",
        )
        plot_cohort_spaghetti(
            dated_df, class_name, args.value_column, args.age_column,
            plots_dir / f"cohort_spaghetti_{class_name}.png",
        )
        plot_cohort_growth_chart(
            dated_df, class_name, args.value_column, args.age_column, args.age_bin_days,
            plots_dir / f"cohort_growth_chart_{class_name}.png",
        )

    plot_summary_boxplot(
        dated_df, args.value_column,
        plots_dir / "summary_boxplot.png",
    )

    print(f"\nDone. Output written to {output_dir}")


if __name__ == "__main__":
    main()