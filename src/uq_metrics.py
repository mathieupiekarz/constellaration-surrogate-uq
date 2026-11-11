"""Measure whether the uncertainty of a trained ensemble is honest (day 4).

No training: reads the per-network predictions saved by `train_baseline.py`
(results/baseline_<split><tag>_predictions.npz) and scores the ensemble's uncertainty.

For each shape, an ensemble of K networks gives
    mu    = mean of the K predictions       -> the prediction
    sigma = standard deviation of the K     -> the uncertainty (disagreement between networks)
and the uncertainty is honest if the true value y falls in mu +/- z * sigma as often as promised.

Metrics, per run, per K and per target (all in z-scored units):
    coverage90 / coverage95  fraction of shapes with |y - mu| <= z * sigma (ideal: 0.90 / 0.95)
    width90                  mean width 2 * 1.645 * sigma of the 90 % interval (smaller is better, if covered)
    nll                      mean Gaussian negative log-likelihood (lower is better; rewards accuracy and honesty)
    z_std                    standard deviation of (y - mu) / sigma (ideal: 1; > 1 means over-confident)
    spearman                 rank correlation between sigma and |y - mu| (does the ensemble hesitate where it errs?)
    miscal_area              mean |observed - expected| coverage over the calibration curve (ideal: 0)

K < 20 is evaluated on disjoint groups of networks (e.g. 6 groups of 3 among 20) and reported as the
mean and the standard deviation over groups, so that the effect of K is not confused with the luck of
which networks were drawn.

Usage:
    uv run python src/uq_metrics.py                              # A, B, paper on the test set
    uv run python src/uq_metrics.py --runs _A_final --ks 3 20    # one run, two ensemble sizes
    uv run python src/uq_metrics.py --set val                    # same on the validation set
"""
import argparse
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

Z90 = NormalDist().inv_cdf(0.95)   # 1.645: mu +/- Z90 * sigma contains 90 % of a Gaussian
Z95 = NormalDist().inv_cdf(0.975)  # 1.960
LEVELS = np.round(np.arange(0.05, 1.0, 0.05), 2)               # calibration curve: 5 %, 10 %, ..., 95 %
Z_LEVELS = np.array([NormalDist().inv_cdf(0.5 + p / 2) for p in LEVELS])


def load_run(results, split, tag, which):
    """Per-network predictions (K, n, targets) and true values (n, targets), z-scored."""
    d = np.load(results / f"baseline_{split}{tag}_predictions.npz")
    targets = [t.replace("metrics.", "") for t in d["targets"]]
    if which == "test":
        return d["pred_z"], d["y_true_z"], targets
    return d["pred_val_z"], d["y_val_z"], targets


def ranks(x):
    """Ranks 0..n-1 along axis 0 (ties are negligible for continuous predictions)."""
    r = np.empty_like(x)
    idx = np.argsort(x, axis=0)
    np.put_along_axis(r, idx, np.arange(x.shape[0])[:, None].astype(x.dtype), axis=0)
    return r


def spearman(a, b):
    """Spearman correlation per column: Pearson correlation of the ranks."""
    ra, rb = ranks(a), ranks(b)
    ra, rb = ra - ra.mean(0), rb - rb.mean(0)
    return (ra * rb).sum(0) / np.sqrt((ra**2).sum(0) * (rb**2).sum(0))


def score(P, y):
    """Uncertainty metrics per target for one ensemble P (K, n, targets) against y (n, targets)."""
    mu = P.mean(axis=0)
    sigma = np.maximum(P.std(axis=0, ddof=1), 1e-8)   # ddof=1: unbiased variance estimate over K networks
    err = y - mu
    z = err / sigma
    observed = (np.abs(z)[None] <= Z_LEVELS[:, None, None]).mean(axis=1)   # (levels, targets)
    metrics = {
        "rmse": np.sqrt((err**2).mean(0)),
        "mean_sigma": sigma.mean(0),
        "coverage90": (np.abs(z) <= Z90).mean(0),
        "coverage95": (np.abs(z) <= Z95).mean(0),
        "width90": (2 * Z90 * sigma).mean(0),
        "nll": (0.5 * np.log(2 * np.pi * sigma**2) + 0.5 * z**2).mean(0),
        "z_std": z.std(0),
        "spearman": spearman(sigma, np.abs(err)),
        "miscal_area": np.abs(observed - LEVELS[:, None]).mean(0),
    }
    return metrics, observed


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", default=["_A_final", "_B_final", "_paper"], help="tags of trained runs")
    p.add_argument("--split", default="RAND")
    p.add_argument("--ks", nargs="+", type=int, default=[3, 5, 10, 20], help="ensemble sizes to evaluate")
    p.add_argument("--set", default="test", choices=["test", "val"], help="shapes on which to score")
    p.add_argument("--results", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    args = p.parse_args()

    rows, curves = [], []
    for tag in args.runs:
        P_all, y, targets = load_run(args.results, args.split, tag, args.set)
        n_members = P_all.shape[0]
        for K in args.ks:
            if K < 2 or K > n_members:
                print(f"skip {tag} K={K}: needs 2 <= K <= {n_members}")
                continue
            for g in range(n_members // K):                     # disjoint groups of K networks
                m, observed = score(P_all[g * K:(g + 1) * K], y)
                for j, t in enumerate(targets):
                    rows.append({"run": tag.lstrip("_"), "K": K, "group": g, "target": t,
                                 **{k: float(v[j]) for k, v in m.items()}})
                    curves += [{"run": tag.lstrip("_"), "K": K, "group": g, "target": t,
                                "expected": float(l), "observed": float(o)}
                               for l, o in zip(LEVELS, observed[:, j])]

    df, cal = pd.DataFrame(rows), pd.DataFrame(curves)
    df.to_csv(args.results / f"uq_metrics_{args.split}_{args.set}.csv", index=False)
    cal.to_csv(args.results / f"uq_calibration_{args.split}_{args.set}.csv", index=False)

    cols = ["rmse", "mean_sigma", "coverage90", "coverage95", "width90", "nll", "z_std", "spearman", "miscal_area"]
    # 1) averaged over the 12 targets, then mean +/- std over the disjoint groups of networks
    per_group = df.groupby(["run", "K", "group"])[cols].mean()
    summary = per_group.groupby(["run", "K"]).agg(["mean", "std"])
    table = pd.DataFrame(index=summary.index)
    for c in cols:
        m, s = summary[(c, "mean")], summary[(c, "std")]
        table[c] = [f"{a:.3f}" if np.isnan(b) else f"{a:.3f}±{b:.3f}" for a, b in zip(m, s)]
    n_groups = per_group.groupby(["run", "K"]).size()
    table.insert(0, "groups", n_groups)
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(f"\n=== {args.set} set, mean over the 12 targets (± std over disjoint groups of K networks) ===")
        print(table.to_string())
        # 2) per target, largest K of each run
        print(f"\n=== per target, K = {max(args.ks)} ===")
        big = df[df.K == df.groupby("run").K.transform("max")]
        print(big.pivot_table(index="target", columns="run", values=["coverage90", "z_std", "spearman"])
                 .round(3).to_string())
    print(f"\nideal: coverage90 0.90, coverage95 0.95, z_std 1.00, miscal_area 0; spearman > 0 means the ensemble"
          f"\nhesitates where it errs. Saved {args.results / f'uq_metrics_{args.split}_{args.set}.csv'}"
          f"\nand the calibration curves in uq_calibration_{args.split}_{args.set}.csv")


if __name__ == "__main__":
    main()
