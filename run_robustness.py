
# ============================================================
# run_robustness.py
# Dropout sweep, missingness sweep, multi-seed.
# Satisfies remarks: 14, 15.
# ============================================================
from __future__ import annotations
import random
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from config import (make_config, Config, DEVICE, VARIANTS, TEST_REPS,
                    ORIGINS, HORIZONS, RESULTS_DIR, SEED)
from data_io import load_truth_obs, prepare, to_tensors
from model import EbolaHybrid
from metrics import rmse, picp
from calibration import apply_factor
from run_experiment import train, mc_forecast


# ============================================================
# Dropout sweep
# ============================================================
def dropout_sweep(variant: str,
                  drop_values=(0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35),
                  seeds=(0, 1, 2),
                  rep: int = 16, origin: int = 90, horizon: int = 7) -> pd.DataFrame:
    print(f"\n=== DROPOUT SWEEP: {variant}  (rep={rep}, origin={origin}, h={horizon}) ===")
    rows = []
    for d in drop_values:
        for s in seeds:
            random.seed(s); np.random.seed(s); torch.manual_seed(s)
            cfg = make_config(variant, rep_for_params=rep)
            cfg.dropout_p = d
            _, obs = load_truth_obs(variant, rep)
            df = prepare(obs, cfg)
            history_df = df.iloc[:origin].copy()
            future_df = df.iloc[origin:origin + horizon].copy()
            hist = to_tensors(history_df, cfg)
            fut = to_tensors(future_df, cfg)
            model = EbolaHybrid(cfg, history_df).to(DEVICE)
            train(model, hist, cfg, verbose=False)
            fc = mc_forecast(model, hist, fut["t"], horizon, cfg)
            y_true = fut["Y"]
            lo, hi = apply_factor(fc["Y_samples"], 1.0)
            rows.append({
                "variant": variant, "dropout": d, "seed": s,
                "RMSE": rmse(y_true, fc["Y_samples"].mean(dim=0)),
                "PICP": picp(y_true, lo, hi),
            })
    df = pd.DataFrame(rows)
    agg = df.groupby("dropout").agg(
        RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
        PICP_mean=("PICP", "mean"), PICP_std=("PICP", "std"),
    ).round(4)
    print(agg.to_string())
    best = float((agg["RMSE_mean"] +
                  10 * (agg["PICP_mean"] - 0.90).abs()).idxmin())
    print(f"[dropout] best (score = RMSE + 10·|PICP - 0.90|): d={best:.2f}")
    df.to_csv(RESULTS_DIR / f"{variant}_dropout_sweep.csv", index=False)
    return df


# ============================================================
# Missingness sweep
# ============================================================
def missingness_sweep(variant: str,
                      levels=(0.0, 0.05, 0.10, 0.20),
                      rep: int = 16, origin: int = 90, horizon: int = 7,
                      cal_factors: dict = None) -> pd.DataFrame:
    print(f"\n=== MISSINGNESS SWEEP: {variant}  (rep={rep}, origin={origin}, h={horizon}) ===")
    rows = []
    for p in levels:
        rng = np.random.default_rng(SEED + int(p * 1000))   # <-- level-specific seed
        cfg = make_config(variant, rep_for_params=rep)
        _, obs = load_truth_obs(variant, rep)
        df = prepare(obs, cfg)
        if p > 0:
            drop = rng.random(len(df)) < p
            df.loc[drop, cfg.cases_col] = np.nan
            df["observed_mask"] = ~drop
            df["logY"] = np.log1p(df[cfg.cases_col].fillna(0.0)).astype(np.float32)

        history_df = df.iloc[:origin].copy()
        future_df = df.iloc[origin:origin + horizon].copy()
        hist = to_tensors(history_df, cfg)
        fut = to_tensors(future_df, cfg)

        if fut["Y"].numel() == 0:                            # <-- empty-tensor guard
            print(f"[missingness] p={p*100:5.1f}%  SKIPPED (empty future window)")
            continue

        model = EbolaHybrid(cfg, history_df).to(DEVICE)
        train(model, hist, cfg, verbose=False)
        fc = mc_forecast(model, hist, fut["t"], horizon, cfg)
        s = cal_factors.get(horizon, 1.0) if cal_factors else 1.0
        lo, hi = apply_factor(fc["Y_samples"], s)

        rmse_val = rmse(fut["Y"], fc["Y_samples"].mean(dim=0))
        if not np.isfinite(rmse_val):                        # <-- second guard
            print(f"[missingness] p={p*100:5.1f}%  SKIPPED (non-finite RMSE)")
            continue

        rows.append({
            "variant": variant, "missingness_pct": p * 100,
            "RMSE": rmse_val,
            "PICP": picp(fut["Y"], lo, hi),
            "Width": float((hi - lo).mean().item()),
        })
        print(f"[missingness] p={p*100:5.1f}%  RMSE={rows[-1]['RMSE']:.2f}  "
              f"PICP={rows[-1]['PICP']:.3f}  Width={rows[-1]['Width']:.2f}")
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / f"{variant}_missingness_sweep.csv", index=False)
    return df


# ============================================================
# Multi-seed run of the main experiment for one variant
# ============================================================
def multi_seed(variant: str, seeds=(0, 1, 2),
               reps=TEST_REPS, origins=ORIGINS, horizons=HORIZONS) -> pd.DataFrame:
    print(f"\n=== MULTI-SEED: {variant} seeds={seeds} ===")
    rows = []
    for s in seeds:
        random.seed(s); np.random.seed(s); torch.manual_seed(s)
        for rep in reps:
            cfg = make_config(variant, rep_for_params=rep)
            _, obs = load_truth_obs(variant, rep)
            df = prepare(obs, cfg)
            for origin in origins:
                if origin >= len(df):
                    continue
                history_df = df.iloc[:origin].copy()
                hist = to_tensors(history_df, cfg)
                model = EbolaHybrid(cfg, history_df).to(DEVICE)
                train(model, hist, cfg, verbose=False)
                for horizon in horizons:
                    if origin + horizon > len(df):
                        continue
                    future_df = df.iloc[origin:origin + horizon].copy()
                    fut = to_tensors(future_df, cfg)
                    fc = mc_forecast(model, hist, fut["t"], horizon, cfg)
                    lo, hi = apply_factor(fc["Y_samples"], 1.0)
                    rows.append({
                        "variant": variant, "seed": s, "rep": rep,
                        "origin": origin, "horizon": horizon,
                        "RMSE": rmse(fut["Y"], fc["Y_samples"].mean(dim=0)),
                        "PICP": picp(fut["Y"], lo, hi),
                    })
    df = pd.DataFrame(rows)
    agg = df.groupby(["horizon"]).agg(
        RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
        PICP_mean=("PICP", "mean"), PICP_std=("PICP", "std"),
    ).round(4)
    print(agg.to_string())
    df.to_csv(RESULTS_DIR / f"{variant}_multi_seed.csv", index=False)
    return df


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "rapid_control"
    variants = VARIANTS if target == "all" else [target]
    for v in variants:
        try:
            dropout_sweep(v)
            # missingness_sweep(v)   # covered by run_missingness.py
            multi_seed(v)
        except Exception as e:
            print(f"[error] {v}: {e}")