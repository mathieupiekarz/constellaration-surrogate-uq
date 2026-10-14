"""Inspect parquet files: columns, types, row counts, missing values and size in bytes.

Works on one file or on a folder of shards (e.g. data/raw/).

Usage:
    uv run python src/inspect_parquet.py data/raw                  # metadata only (fast)
    uv run python src/inspect_parquet.py data/raw --full           # + missing values, memory, preview
    uv run python src/inspect_parquet.py data/processed/constellaration.parquet --full --head 3
"""

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def human(n_bytes: float) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


def list_files(path: Path) -> list[Path]:
    files = [path] if path.is_file() else sorted(path.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No .parquet file found at {path}")
    return files


def inspect_metadata(files: list[Path]) -> None:
    """Reads only the file footers: instant, even for GB-sized files."""
    print("=== Files ===")
    total_rows = total_disk = total_raw = 0
    schemas = set()
    for f in files:
        meta = pq.ParquetFile(f).metadata
        disk = f.stat().st_size
        raw = sum(meta.row_group(i).total_byte_size for i in range(meta.num_row_groups))
        total_rows += meta.num_rows
        total_disk += disk
        total_raw += raw
        schemas.add(str(pq.read_schema(f)))
        print(f"{f.name}: {meta.num_rows:>8} rows | {meta.num_columns} columns | "
              f"{disk:>12,} B on disk ({human(disk)}) | ~{human(raw)} uncompressed")
    print(f"TOTAL: {total_rows} rows | {total_disk:,} B on disk ({human(total_disk)}) | "
          f"~{human(total_raw)} uncompressed (compression x{total_raw / total_disk:.1f})")
    print("Same schema in every file:", "yes" if len(schemas) == 1 else f"NO ({len(schemas)} different)")

    print("\n=== Columns (name : type) ===")
    for field in pq.read_schema(files[0]):
        print(f"  {field.name} : {field.type}")


def inspect_content(files: list[Path], head: int) -> None:
    """Loads the data in memory: missing values, memory size, preview."""
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"\n=== Content ===\nshape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"memory once loaded in pandas: {human(df.memory_usage(deep=True).sum())}")

    na = pd.DataFrame({"n_missing": df.isna().sum(), "pct_missing": (df.isna().mean() * 100).round(1)})
    na = na[na.n_missing > 0].sort_values("n_missing", ascending=False)
    print(f"\ncolumns with missing values: {len(na)} / {df.shape[1]}")
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(na.to_string() if len(na) else "  none")

    if head:
        print(f"\n=== First {head} rows (one column per line) ===")
        def short(v, n=70):  # matrices are long: keep only the beginning
            s = str(v).replace("\n", " ")
            return s if len(s) <= n else s[:n] + " ..."
        preview = df.head(head).T.apply(lambda col: col.map(short))
        with pd.option_context("display.max_rows", None, "display.width", 200):
            print(preview.to_string())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="a .parquet file or a folder of .parquet files")
    parser.add_argument("--full", action="store_true", help="also load the data (slower)")
    parser.add_argument("--head", type=int, default=2, help="rows to preview with --full")
    args = parser.parse_args()
    files = list_files(Path(args.path))
    inspect_metadata(files)
    if args.full:
        inspect_content(files, args.head)


if __name__ == "__main__":
    main()
