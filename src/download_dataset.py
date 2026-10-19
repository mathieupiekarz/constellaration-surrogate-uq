"""Download the ConStellaration 'default' subset and build ML-ready tables.

Layout produced (everything under data/, which is git-ignored):
    data/raw/train-0000X-of-00003.parquet        untouched Hugging Face shards
    data/processed/constellaration.parquet       all rows, flattened Fourier coefficients
    data/sample/constellaration_sample.parquet   random subsample for fast experiments

Usage (from the project root):
    uv run python src/download_dataset.py                 # download + process
    uv run python src/download_dataset.py --n-sample 5000
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

REPO_ID = "proxima-fusion/constellaration"
SEED = 42
FOURIER_COLS = ["boundary.r_cos", "boundary.z_sin"]
METHOD_ID_COLS = {
    "desc": "desc_omnigenous_field_optimization_settings.id",
    "vmec": "vmec_omnigenous_field_optimization_settings.id",
    "qp_init": "qp_init_omnigenous_field_optimization_settings.id",
    "nae_init": "nae_init_omnigenous_field_optimization_settings.id",
}


def download_raw(raw_dir: Path) -> list[Path]:
    """Download only the 'default' shards (~610 MB) straight into raw_dir.

    snapshot_download mirrors the repo layout (files live under 'data/' on the Hub), so we
    download into a temporary folder, move the shards to raw_dir, then delete the folder.
    Skips the download if shards are already in raw_dir.
    """
    files = sorted(raw_dir.glob("train-*.parquet"))
    if files:
        print(f"  {len(files)} shards already in {raw_dir}, download skipped")
        return files
    tmp = raw_dir / "_download_tmp"
    snapshot_download(repo_id=REPO_ID, repo_type="dataset",
                      allow_patterns=["data/*.parquet"], local_dir=str(tmp))
    for f in (tmp / "data").glob("*.parquet"):
        shutil.move(str(f), str(raw_dir / f.name))
    shutil.rmtree(tmp, ignore_errors=True)
    files = sorted(raw_dir.glob("train-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet shard found in {raw_dir}")
    return files


def load_raw(files: list[Path]) -> pd.DataFrame:
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    expected = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    assert len(df) == expected, f"row count mismatch: {len(df)} vs {expected}"
    return df


def to_matrix(cell) -> np.ndarray | None:
    """Parquet stores a 2D list as an array of arrays; turn it into a 2D float matrix."""
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return None
    return np.stack([np.asarray(row, dtype=float) for row in cell])


def flatten_fourier(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """r_cos[m, j] -> columns 'r_cos_m{m}_n{n}' with n = j - n_tor (n from -n_tor to +n_tor).

    Matrices of different sizes are zero-padded to the largest (m, n) seen in the data,
    which is exact: a missing high-order mode is a zero coefficient.
    """
    mats = [to_matrix(c) for c in df[col]]
    shapes = {m.shape for m in mats if m is not None}
    max_m = max(s[0] for s in shapes)
    max_ntor = max((s[1] - 1) // 2 for s in shapes)
    name = col.split(".")[1]
    out = np.full((len(df), max_m, 2 * max_ntor + 1), np.nan)
    for i, mat in enumerate(mats):
        if mat is None:
            continue
        out[i] = 0.0
        ntor = (mat.shape[1] - 1) // 2
        out[i, : mat.shape[0], max_ntor - ntor: max_ntor + ntor + 1] = mat
    cols = [f"{name}_m{m}_n{n}" for m in range(max_m) for n in range(-max_ntor, max_ntor + 1)]
    print(f"  {col}: shapes found {sorted(shapes)} -> {len(cols)} columns")
    return pd.DataFrame(out.reshape(len(df), -1), columns=cols, index=df.index)


def build_table(df: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in df.columns
            if (c.startswith("metrics.") or c.startswith("boundary.") or c.startswith("misc."))
            and not c.endswith(".json") and c not in FOURIER_COLS]
    keep += [c for c in ["plasma_config_id"] if c in df.columns]
    table = df[keep].copy()

    # generation method (needed to reproduce the paper's baseline filter, appendix A.4)
    table["method"] = "unknown"
    for name, col in METHOD_ID_COLS.items():
        if col in df.columns:
            table.loc[df[col].notna() & (table["method"] == "unknown"), "method"] = name

    err_cols = [c for c in table.columns if c.startswith("misc.has_") and c.endswith("_error")]
    table["has_any_error"] = table[err_cols].fillna(False).astype(bool).any(axis=1)

    # r_sin / z_cos are empty for stellarator-symmetric boundaries: drop them only if truly empty
    for c in ["boundary.r_sin", "boundary.z_cos"]:
        if c in table.columns and table[c].isna().all():
            table = table.drop(columns=c)
        elif c in table.columns:
            print(f"  WARNING: {c} is not always empty, kept as is")

    fourier = [flatten_fourier(df, c) for c in FOURIER_COLS]
    return pd.concat([table] + fourier, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--n-sample", type=int, default=20000)
    args = parser.parse_args()
    data = Path(args.data_dir)
    for sub in ["raw", "processed", "sample"]:
        (data / sub).mkdir(parents=True, exist_ok=True)

    print("1/3 Downloading raw shards...")
    files = download_raw(data / "raw")
    for f in files:
        print(f"  {f.name}: {f.stat().st_size / 1e6:.0f} MB")
    df = load_raw(files)
    print(f"  raw table: {df.shape[0]} rows x {df.shape[1]} columns")

    print("2/3 Building processed table...")
    table = build_table(df)
    table.to_parquet(data / "processed" / "constellaration.parquet", index=False)
    print(f"  processed: {table.shape} -> data/processed/constellaration.parquet")
    print("  rows per n_field_periods:\n" + table["boundary.n_field_periods"].value_counts().sort_index().to_string())
    print("  rows per method:\n" + table["method"].value_counts().to_string())
    print(f"  rows with any error flag: {int(table['has_any_error'].sum())}")
    na = table[[c for c in table.columns if c.startswith("metrics.")]].isna().mean().mul(100).round(1)
    print("  % missing per metric:\n" + na.to_string())

    print("3/3 Sampling...")
    sample = table.sample(n=min(args.n_sample, len(table)), random_state=SEED)
    sample.to_parquet(data / "sample" / "constellaration_sample.parquet", index=False)
    print(f"  sample: {sample.shape} -> data/sample/constellaration_sample.parquet")


if __name__ == "__main__":
    main()
