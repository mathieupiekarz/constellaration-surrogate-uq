"""Reproduce the paper's surrogate: an ensemble of MLPs mapping boundary Fourier coefficients to the 12 metrics.

Protocol (ConStellaration paper, Appendix A.4): paper baseline subset (NFP = 3, DESC/VMEC, 0.05 % tails
trimmed), 20 % held out for testing, targets z-scored with training statistics, ensemble of 10 MLPs with
3 layers of 256 units and tanh activations, mean squared error loss.

Choices not specified by the paper (stated in the README):
  * "3 layers" is read as 3 hidden layers of 256 units;
  * inputs are z-scored with training statistics as well;
  * Adam (AdamW when weight decay > 0), learning rate 1e-3 halved when the validation loss stalls for 10 epochs, batch size 256,
    early stopping on a validation set (10 % of the training set);
  * ensemble members differ only by their random seed (initialisation and batch order), not by the data;
  * optional regularisation against overfitting: weight decay (AdamW) and dropout after each hidden layer.
    Hyperparameters are compared on the VALIDATION score only; the test set is reported, never used to choose.

Outputs (in --out, default results/):
  baseline_<split><tag>_metrics.csv       per-target RMSE (z-scored and physical units) and R2 of the ensemble mean
  baseline_<split><tag>_predictions.npz   per-member test predictions, reused later for the uncertainty metrics
  baseline_<split><tag>_training_log.json settings, per-member history, ensemble train/validation MSE

Usage:
    uv run python src/train_baseline.py                      # RAND split, 10 members
    uv run python src/train_baseline.py --n-members 2 --epochs 20   # quick check
    uv run python src/train_baseline.py --weight-decay 1e-4 --dropout 0.1 --tag _wd1e-4_do0.1
    uv run python src/train_baseline.py --threads 2                   # leave CPU cores free for other work
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

import data as D


class MLP(nn.Module):
    def __init__(self, n_in: int, n_out: int, width: int = 256, depth: int = 3, dropout: float = 0.0):
        super().__init__()
        layers, d = [], n_in
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.Tanh()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = width
        layers.append(nn.Linear(d, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def split_rand(base: pd.DataFrame, seed: int, test_frac: float = 0.2) -> tuple[pd.Index, pd.Index]:
    """Random split, identical to the RAND split of notebook 02 for the same seed."""
    rng = np.random.default_rng(seed)
    test = base.index[rng.permutation(len(base))[: int(test_frac * len(base))]]
    return base.index.difference(test), test


def train_member(X_tr, y_tr, X_val, y_val, seed, epochs, patience, batch_size, lr, schedule="plateau",
                 width=256, depth=3, dropout=0.0, weight_decay=0.0):
    """Train one MLP with early stopping on the validation loss; returns the best model and its history.

    AdamW with weight_decay = 0 is the same algorithm as Adam."""
    torch.manual_seed(seed)
    model = MLP(X_tr.shape[1], y_tr.shape[1], width=width, depth=depth, dropout=dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10) if schedule == "plateau" else None
    loss_fn = nn.MSELoss()
    gen = torch.Generator().manual_seed(seed)
    best, best_state, wait, history = np.inf, None, 0, []
    for epoch in range(epochs):
        model.train()
        for idx in torch.randperm(len(X_tr), generator=gen).split(batch_size):
            opt.zero_grad()
            loss = loss_fn(model(X_tr[idx]), y_tr[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = loss_fn(model(X_val), y_val).item()
        history.append(val)
        if sched is not None:
            sched.step(val)
        if val < best - 1e-6:
            best, wait = val, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return model, history


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="RAND", choices=["RAND"])
    p.add_argument("--n-members", type=int, default=10)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--schedule", default="plateau", choices=["plateau", "none"],
                   help="plateau: halve the learning rate when the validation loss stalls for 10 epochs")
    p.add_argument("--width", type=int, default=256, help="units per hidden layer")
    p.add_argument("--depth", type=int, default=3, help="number of hidden layers")
    p.add_argument("--dropout", type=float, default=0.0, help="dropout rate after each hidden layer (0 = off)")
    p.add_argument("--weight-decay", type=float, default=0.0, help="AdamW weight decay (0 = plain Adam)")
    p.add_argument("--threads", type=int, default=0, help="CPU threads used by PyTorch (0 = PyTorch default)")
    p.add_argument("--tag", default="", help="suffix added to the output file names, e.g. _wd1e-4")
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    name = f"baseline_{args.split}{args.tag}"

    base = D.paper_baseline_subset(D.clean(D.load_processed()))
    train_idx, test_idx = split_rand(base, args.seed)
    rng = np.random.default_rng(args.seed + 1)
    val_idx = train_idx[rng.permutation(len(train_idx))[: int(0.1 * len(train_idx))]]
    fit_idx = train_idx.difference(val_idx)
    print(f"split {args.split}: fit {len(fit_idx)} | validation {len(val_idx)} | test {len(test_idx)}")

    features = D.feature_columns(base.loc[fit_idx])
    x_mu, x_sd = base.loc[fit_idx, features].mean(), base.loc[fit_idx, features].std()
    y_mu, y_sd = base.loc[fit_idx, D.TARGETS].mean(), base.loc[fit_idx, D.TARGETS].std()

    def tensors(idx):
        X = ((base.loc[idx, features] - x_mu) / x_sd).to_numpy(np.float32, copy=True)
        y = ((base.loc[idx, D.TARGETS] - y_mu) / y_sd).to_numpy(np.float32, copy=True)
        return torch.from_numpy(X), torch.from_numpy(y)

    (X_fit, y_fit), (X_val, y_val), (X_te, y_te) = tensors(fit_idx), tensors(val_idx), tensors(test_idx)
    print(f"{len(features)} input features, {len(D.TARGETS)} targets")

    preds, preds_val, preds_fit, log = [], [], [], []
    for k in range(args.n_members):
        t0 = time.time()
        model, hist = train_member(X_fit, y_fit, X_val, y_val, args.seed + 100 + k,
                                   args.epochs, args.patience, args.batch_size, args.lr, args.schedule,
                                   args.width, args.depth, args.dropout, args.weight_decay)
        model.eval()
        with torch.no_grad():
            preds.append(model(X_te).numpy())
            preds_val.append(model(X_val).numpy())
            preds_fit.append(model(X_fit).numpy())
        log.append({"member": k, "epochs": len(hist), "best_val_mse": float(min(hist)), "seconds": time.time() - t0})
        print(f"  member {k}: {len(hist)} epochs, best validation MSE {min(hist):.4f}, {time.time() - t0:.0f} s")

    val_mse = float(np.mean((np.mean(preds_val, axis=0) - y_val.numpy())**2))   # ensemble, all targets
    fit_mse = float(np.mean((np.mean(preds_fit, axis=0) - y_fit.numpy())**2))   # same, on the training rows
    P = np.stack(preds)                    # (members, n_test, targets), z-scored
    mean = P.mean(axis=0)
    yt = y_te.numpy()
    sd = y_sd.to_numpy()
    rows = []
    for j, t in enumerate(D.TARGETS):
        err = mean[:, j] - yt[:, j]
        rows.append({"target": t.replace("metrics.", ""),
                     "rmse_zscored": np.sqrt(np.mean(err**2)),
                     "rmse_physical": np.sqrt(np.mean(err**2)) * sd[j],
                     "r2": 1 - np.sum(err**2) / np.sum((yt[:, j] - yt[:, j].mean())**2),
                     "rmse_single_member_mean": np.mean([np.sqrt(np.mean((P[m, :, j] - yt[:, j])**2)) for m in range(len(P))])})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.out / f"{name}_metrics.csv", index=False)
    np.savez_compressed(args.out / f"{name}_predictions.npz",
                        pred_z=P, y_true_z=yt, y_mu=y_mu.to_numpy(), y_sd=sd,
                        test_index=np.asarray(test_idx), targets=np.array(D.TARGETS))
    (args.out / f"{name}_training_log.json").write_text(json.dumps(
        {"args": {k: str(v) for k, v in vars(args).items()}, "ensemble_train_mse": fit_mse,
         "ensemble_val_mse": val_mse, "members": log}, indent=1))

    with pd.option_context("display.width", 160, "display.float_format", "{:.4f}".format):
        print(metrics.to_string(index=False))
    print(f"\nensemble MSE (all targets, z-scored): train {fit_mse:.4f} | validation {val_mse:.4f}"
          f" | gap x{val_mse / fit_mse:.1f}   <- compare settings on the validation MSE")
    print(f"all targets: R2 min {metrics.r2.min():.4f} | RMSE (z-scored) max {metrics.rmse_zscored.max():.4f}"
          f"   (paper: R2 > 0.97, RMSE < 0.1 for all metrics)")


if __name__ == "__main__":
    main()
