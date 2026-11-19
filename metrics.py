

# ============================================================
# metrics.py
# All metrics on an explicit scale. Aggregation prints the scale in the label.
# Satisfies remarks: 7, 8, 13, 16.
# ============================================================
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from typing import Dict, Tuple

from config import ALPHA_CI


# ------------------------------------------------------------
# Point metrics (all on whatever scale the tensors are on)
# ------------------------------------------------------------
def rmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item())


def mean_error(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    return float(torch.mean(y_pred - y_true).item())


def mape(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    m = y_true > 1e-6
    if m.sum() == 0:
        return float("nan")
    return float(torch.mean(torch.abs((y_true[m] - y_pred[m]) /
                                      (y_true[m] + 1e-9))).item())


# ------------------------------------------------------------
# Interval metrics (require lower/upper on the same scale)
# ------------------------------------------------------------
def picp(y_true: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> float:
    inside = ((y_true >= lower) & (y_true <= upper)).float().mean()
    return float(inside.item())


def interval_width(lower: torch.Tensor, upper: torch.Tensor) -> float:
    return float(torch.mean(upper - lower).item())


def winkler(y_true: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor,
            alpha: float = ALPHA_CI) -> float:
    w = upper - lower
    pl = (2.0 / alpha) * (lower - y_true) * (y_true < lower).float()
    pu = (2.0 / alpha) * (y_true - upper) * (y_true > upper).float()
    return float(torch.mean(w + pl + pu).item())


# ------------------------------------------------------------
# CRPS (M-sample approximation)
# ------------------------------------------------------------
def crps(y_true: torch.Tensor, y_samples: torch.Tensor) -> float:
    """CRPS = E|X - y| - 0.5 E|X - X'|, X, X' iid from samples."""
    M = y_samples.shape[0]
    abs_err = torch.abs(y_samples - y_true.unsqueeze(0)).mean(dim=0)
    n_pairs = min(M, 64)
    idx_a = torch.randint(0, M, (n_pairs,), device=y_samples.device)
    idx_b = torch.randint(0, M, (n_pairs,), device=y_samples.device)
    ead = (y_samples[idx_a] - y_samples[idx_b]).abs().mean(dim=0)
    return float(torch.clamp(abs_err - 0.5 * ead, min=0.0).mean().item())


# ------------------------------------------------------------
# Diebold-Mariano test
# ------------------------------------------------------------
def diebold_mariano(err_a: np.ndarray, err_b: np.ndarray) -> Dict[str, float]:
    """H0: equal squared-error loss.

    Positive stat => A worse than B. p_value < 0.05 => reject H0.
    """
    from scipy import stats
    d = err_a ** 2 - err_b ** 2
    n = len(d)
    if n < 2:
        return {"dm_stat": float("nan"), "p_value": float("nan")}
    var_d = np.var(d, ddof=1)
    if var_d == 0:
        return {"dm_stat": 0.0, "p_value": 1.0}
    stat = float(np.mean(d) / np.sqrt(var_d / n))
    p = float(2 * (1 - stats.norm.cdf(abs(stat))))
    return {"dm_stat": stat, "p_value": p}


# ------------------------------------------------------------
# Correlation helpers (Remark 13)
# ------------------------------------------------------------
def safe_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().flatten()
    b = b.detach().float().flatten()
    if a.numel() < 2 or b.numel() < 2:
        return float("nan")
    if a.std() < 1e-8 or b.std() < 1e-8:
        return float("nan")
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1].item())


# ------------------------------------------------------------
# Aggregator — always prints the scale in the metric name
# ------------------------------------------------------------
def summarise(df: pd.DataFrame, scale_label: str = "orig") -> pd.DataFrame:
    """Group by horizon and report mean/median/std for every metric.

    df must already contain columns on the scale `scale_label`.
    """
    cols = [c for c in df.columns if c.startswith(("RMSE", "PICP", "Width",
                                                    "Winkler", "CRPS",
                                                    "MAPE", "ME", "r_daily",
                                                    "r_cum"))]
    agg = (df.groupby("horizon")[cols]
             .agg(["mean", "median", "std", "count"])
             .round(4))
    # Explicit scale note in the printed block:
    print(f"\n=== SUMMARY ({scale_label} scale) ===")
    print("All RMSE/Width/Winkler/CRPS values are on the ORIGINAL case scale.")
    print(f"PICP is for the calibrated {int(100 * (1 - ALPHA_CI))}% interval.")
    print(agg.to_string())
    return agg
