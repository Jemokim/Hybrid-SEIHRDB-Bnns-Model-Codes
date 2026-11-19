
# ============================================================
# calibration.py
# Fit s on reps 11–15 ONLY, per (horizon, series). Freeze. Apply.
# Satisfies remarks: 3, 4.
# ============================================================
from __future__ import annotations
import numpy as np
import torch
from typing import Dict, Tuple

from config import Config, LOWER_Q, UPPER_Q


# ------------------------------------------------------------
# Single-window fit
# ------------------------------------------------------------
def fit_s_one_window(y_true: torch.Tensor, y_samples: torch.Tensor,
                     target: float = 0.90,
                     lo_q: float = LOWER_Q, hi_q: float = UPPER_Q,
                     s_min: float = 0.50, s_max: float = 2.0) -> float:
    """Return scalar s such that scaling the raw quantiles reaches target PICP."""
    if y_true.numel() == 0 or y_samples.numel() == 0:
        return 1.0
    if y_samples.dim() == 1:
        y_samples = y_samples.unsqueeze(0)
    if y_samples.shape[1] != y_true.numel():
        # nothing we can do; return neutral
        return 1.0

    y_mean = y_samples.mean(dim=0)
    lo0 = torch.quantile(y_samples, lo_q, dim=0)
    hi0 = torch.quantile(y_samples, hi_q, dim=0)

    def picp_at(s):
        lo = y_mean - s * (y_mean - lo0)
        hi = y_mean + s * (hi0 - y_mean)
        return float(((y_true >= lo) & (y_true <= hi)).float().mean().item())

    # If raw intervals already cover, shrink from 1.0 downward
    if picp_at(1.0) >= target:
        s = 1.0
        for _ in range(60):
            s *= 0.95
            if s < s_min:
                return s_min
            if picp_at(s) < target:
                return float(np.clip(s * 1.02, s_min, s_max))
        return 1.0

    # Otherwise expand upward from 1.0
    s = 1.0
    for _ in range(40):
        s *= 1.05
        if picp_at(s) >= target or s >= s_max:
            break
    return float(np.clip(s, s_min, s_max))


# ------------------------------------------------------------
# Freeze over validation replicates (reps 11–15) and (origin, horizon) grid
# ------------------------------------------------------------
def fit_calibration_factors(
    variant: str,
    origins: Tuple[int, ...],
    horizons: Tuple[int, ...],
    predict_fn,            # callable(history, horizon, cfg) -> {"Y_samples": ...}
    train_fn,              # callable(df, cfg, origin) -> trained model
    load_and_prepare,      # callable(variant, rep, cfg) -> df
    cfg_factory,           # callable(variant) -> Config
    val_reps: Tuple[int, ...] = (11, 12, 13, 14, 15),
    verbose: bool = True,
) -> Dict[int, float]:
    """Fit one s per horizon by taking the median over all val windows.

    Returns {horizon: s_frozen}.
    """
    per_h: Dict[int, list] = {h: [] for h in horizons}

    for rep in val_reps:
        cfg = cfg_factory(variant)
        df = load_and_prepare(variant, rep, cfg)
        for origin in origins:
            if origin >= len(df):
                continue
            history_df = df.iloc[:origin].copy()
            model = train_fn(df, cfg, origin)

            for horizon in horizons:
                if origin + horizon > len(df):
                    continue
                future_df = df.iloc[origin:origin + horizon].copy()
                fc = predict_fn(model, history_df, future_df, horizon, cfg)
                y_true = fc["Y_true"]
                y_samples = fc["Y_samples"]
                s = fit_s_one_window(y_true, y_samples, target=cfg.target_picp)
                per_h[horizon].append(s)
                if verbose:
                    print(f"[cal] rep={rep:02d} origin={origin:3d} "
                          f"h={horizon:2d}  s={s:.3f}")

    frozen: Dict[int, float] = {}
    for h in horizons:
        vals = per_h[h]
        med = float(np.median(vals)) if vals else 1.0
        frozen[h] = float(np.clip(med, 0.50, 2.0))
        if verbose:
            print(f"[cal] h={h:2d}  s values = {[round(v, 3) for v in vals]}")
            print(f"[cal] h={h:2d}  median = {frozen[h]:.3f}  (n={len(vals)})")

    print(f"[cal] FROZEN calibration for {variant}: {frozen}")
    return frozen


# ------------------------------------------------------------
# Apply a frozen factor
# ------------------------------------------------------------
def apply_factor(y_samples: torch.Tensor, s: float,
                 lo_q: float = LOWER_Q, hi_q: float = UPPER_Q
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    if y_samples.numel() == 0:
        z = torch.empty(0, device=y_samples.device if torch.is_tensor(y_samples) else "cpu")
        return z, z
    if y_samples.dim() == 1:
        y_samples = y_samples.unsqueeze(0)
    y_mean = y_samples.mean(dim=0)
    lo0 = torch.quantile(y_samples, lo_q, dim=0)
    hi0 = torch.quantile(y_samples, hi_q, dim=0)
    lo = y_mean - s * (y_mean - lo0)
    hi = y_mean + s * (hi0 - y_mean)
    return lo, hi