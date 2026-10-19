"""Shared data loading and filtering, so that every notebook and script uses the same cleaning.

Reads data/processed/constellaration.parquet (built by src/download_dataset.py).
Run `uv run python src/data.py` to save the filtered paper-baseline subset to
data/processed/constellaration_filtered.parquet.

Choices made here (all documented in the README / report):
  * rows without metrics are dropped (the physics code was not run or failed);
  * qi spans several orders of magnitude, so the target is log10(qi) (our choice, to be stated);
  * model inputs are the flattened Fourier coefficients only (r_cos_*, z_sin_*); columns that are
    constant over the selected rows (e.g. modes fixed to 0 by symmetry) are removed;
  * the paper baseline subset follows the ConStellaration paper (arXiv:2506.19583, Appendix A.4):
    NFP = 3, configurations from the DESC or VMEC optimizations, 0.05 % tails removed per metric.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "constellaration.parquet"

NFP = "boundary.n_field_periods"
METHOD = "method"

# the 12 metrics computed by constellaration/forward_model.py (ConstellarationMetrics);
# qi is replaced by log10(qi)
TARGETS = [
    "metrics.aspect_ratio",
    "metrics.aspect_ratio_over_edge_rotational_transform",
    "metrics.max_elongation",
    "metrics.axis_rotational_transform_over_n_field_periods",
    "metrics.edge_rotational_transform_over_n_field_periods",
    "metrics.axis_magnetic_mirror_ratio",
    "metrics.edge_magnetic_mirror_ratio",
    "metrics.average_triangularity",
    "metrics.vacuum_well",
    "metrics.minimum_normalized_magnetic_gradient_scale_length",
    "metrics.flux_compression_in_regions_of_bad_curvature",
    "metrics.log10_qi",
]


def load_processed(path: Path = DEFAULT_PATH) -> pd.DataFrame:
    return pd.read_parquet(path)


def add_log10_qi(df: pd.DataFrame) -> pd.DataFrame:
    """log10(qi); qi <= 0 (impossible for a sum of squares, but checked) becomes NaN."""
    qi = df["metrics.qi"]
    out = df.copy()
    out["metrics.log10_qi"] = np.log10(qi.where(qi > 0))
    return out


def clean(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Drop rows without any metric, then add log10(qi)."""
    metric_cols = [c for c in df.columns if c.startswith("metrics.")]
    has_metrics = df[metric_cols].notna().any(axis=1)
    out = add_log10_qi(df[has_metrics])
    if verbose:
        print(f"clean: {len(df)} rows -> {len(out)} with metrics ({(~has_metrics).sum()} dropped)")
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Flattened Fourier coefficients that actually vary over the rows of df."""
    cols = [c for c in df.columns if c.startswith(("r_cos_", "z_sin_"))]
    return [c for c in cols if df[c].nunique(dropna=True) > 1]


def trim_tails(df: pd.DataFrame, cols: list[str], tail: float = 0.0005) -> pd.DataFrame:
    """Keep rows inside [quantile(tail), quantile(1 - tail)] for EVERY column of cols.

    Interpretation of the paper's '0.05 % tails for each of the metrics': 0.05 % cut at
    each end of each metric, quantiles computed on the subset before any cut.
    """
    keep = pd.Series(True, index=df.index)
    for c in cols:
        lo, hi = df[c].quantile([tail, 1 - tail])
        keep &= df[c].between(lo, hi)
    return df[keep]


def paper_baseline_subset(df: pd.DataFrame, tail: float = 0.0005, verbose: bool = True) -> pd.DataFrame:
    """Subset used for the paper's MLP-ensemble surrogate (Appendix A.4). Expects clean(df)."""
    steps = [("cleaned input", df)]
    sub = df[df[NFP] == 3]
    steps.append(("NFP = 3", sub))
    sub = sub[sub[METHOD].isin(["desc", "vmec"])]
    steps.append(("DESC or VMEC", sub))
    sub = sub.dropna(subset=TARGETS)
    steps.append(("all 12 targets present", sub))
    sub = trim_tails(sub, TARGETS, tail)
    steps.append((f"tails {tail:.3%} trimmed per target", sub))
    if verbose:
        for name, d in steps:
            print(f"  {name:32s}: {len(d):>7} rows")
    return sub


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean and filter the ConStellaration dataset.")
    parser.add_argument("--input", type=Path, default=DEFAULT_PATH, help="input processed parquet")
    parser.add_argument(
        "--output",
        type=Path,
        help="output parquet (defaults to <input>_filtered.parquet beside the input)",
    )
    parser.add_argument("--tail", type=float, default=0.0005, help="tail fraction trimmed per target")
    args = parser.parse_args()

    cleaned = clean(load_processed(args.input))
    filtered = paper_baseline_subset(cleaned, tail=args.tail)
    output_path = args.output or args.input.with_name(f"{args.input.stem}_filtered.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_parquet(output_path, index=False)
    print(f"saved: {filtered.shape} -> {output_path}")


if __name__ == "__main__":
    main()
