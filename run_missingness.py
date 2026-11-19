# ============================================================
# run_missingness.py
# Controlled missing-data experiment (Remark 15).
#
# Compares forecasting performance at 0%, 5%, 10%, 20% additional
# missing-at-random observations on top of the generator's own
# missingness (already encoded in observed_mask by data_io.prepare).
#
# For each variant, each test replicate, each (origin, horizon):
#   1. Load the prepared dataframe (with generator's mask)
#   2. For the target level, additionally flip observed_mask=False
#      at randomly chosen history indices
#   3. Fit the BNN on the masked history
#   4. Forecast the same future window (unmasked)
#   5. Record RMSE, PICP, width, CRPS on the original scale
#
# Output: results/missingness_results.csv
# ============================================================
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
import torch

from config import (reset_seed, SEED, make_config, VARIANTS, TEST_REPS,
                    ORIGINS, HORIZONS, RESULTS_DIR)
from data_io import load_truth_obs, prepare, to_tensors
from model import EbolaHybrid
from metrics import rmse, picp, crps
from calibration import apply_factor, fit_calibration_factors
from run_experiment import train, mc_forecast


# ------------------------------------------------------------
# Missingness injection
# ------------------------------------------------------------
def add_missingness_mask(df: pd.DataFrame, target_fraction: float,
                         seed: int) -> pd.DataFrame:
    """Return a copy of df with `target_fraction` additional entries
    flipped to observed_mask = False in the history window.

    The generator's own missingness is preserved. The additional
    masking is applied uniformly at random over the history window.
    """
    df = df.copy()
    rng = np.random.default_rng(seed)
    n = len(df)
    n_extra = int(round(n * target_fraction))
    if n_extra > 0:
        idx = rng.choice(n, size=n_extra, replace=False)
        df.loc[df.index[idx], "observed_mask"] = False
    return df


# ------------------------------------------------------------
# One variant × one missingness level
# ------------------------------------------------------------
def evaluate_variant_level(variant: str, target_fraction: float,
                           s_frozen: dict, verbose: bool = True) -> pd.DataFrame:
    reset_seed(SEED)
    rows = []
    level_pct = int(round(target_fraction * 100))
    print(f"\n=== Missingness: {variant} @ target +{level_pct}% ===")

    for rep in TEST_REPS:
        cfg = make_config(variant, rep_for_params=rep)
        _, obs = load_truth_obs(variant, rep)
        df_full = prepare(obs, cfg)
        scale = float(df_full["scale_factor"].iloc[0])

        for origin in ORIGINS:
            if origin >= len(df_full):
                continue

            # History window up to origin
            hist_base = df_full.iloc[:origin].copy()

            # Add controlled missingness to history only
            hist_masked = add_missingness_mask(
                hist_base, target_fraction, seed=SEED + rep * 100 + origin
            )

            # Fit model on masked history
            history = to_tensors(hist_masked, cfg)
            model = EbolaHybrid(cfg, hist_masked).to(history["Y"].device)
            train(model, history, cfg, verbose=False)

            for horizon in HORIZONS:
                if origin + horizon > len(df_full):
                    continue
                future_df = df_full.iloc[origin:origin + horizon].copy()
                future = to_tensors(future_df, cfg)

                fc = mc_forecast(model, history, future["t"], horizon, cfg)
                y_true = future["Y"]
                y_samples = fc["Y_samples"]

                # Frozen calibration + non-negativity floor
                lo, hi = apply_factor(y_samples, s_frozen[horizon])
                lo = torch.minimum(lo, torch.zeros_like(lo))

                y_mean = y_samples.mean(dim=0)

                rows.append({
                    "variant": variant,
                    "target_missingness": target_fraction,
                    "rep": rep,
                    "origin": origin,
                    "horizon": horizon,
                    "RMSE_orig": rmse(y_true, y_mean) / scale,
                    "PICP": picp(y_true, lo, hi),
                    "Width_orig": float((hi - lo).mean().item()) / scale,
                    "CRPS_orig": crps(y_true, y_samples) / scale,
                })

                if verbose:
                    r = rows[-1]
                    print(f"  rep={rep} origin={origin:3d} h={horizon:2d}  "
                          f"RMSE={r['RMSE_orig']:.2f}  "
                          f"PICP={r['PICP']:.3f}  "
                          f"Width={r['Width_orig']:.2f}")

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def run_all(target_variants=None, levels=(0.0, 0.05, 0.10, 0.20)):
    if target_variants is None:
        target_variants = VARIANTS

    # --- Load existing results so re-runs append instead of overwrite ---
    combined_path = RESULTS_DIR / "missingness_results.csv"
    if combined_path.exists():
        existing = pd.read_csv(combined_path)
        # Drop rows for any variant we're about to re-run
        existing = existing[~existing["variant"].isin(target_variants)]
        all_rows = [existing] if len(existing) else []
    else:
        all_rows = []

    for variant in target_variants:
        print(f"\n{'=' * 70}")
        print(f"MISSINGNESS EXPERIMENT: {variant}")
        print(f"{'=' * 70}")

        from run_experiment import _load_and_prepare, _train_fn, _predict_fn
        s_frozen = fit_calibration_factors(
            variant=variant,
            origins=ORIGINS,
            horizons=HORIZONS,
            predict_fn=_predict_fn,
            train_fn=_train_fn,
            load_and_prepare=_load_and_prepare,
            cfg_factory=lambda v: make_config(v, rep_for_params=11),
            val_reps=(11, 12, 13, 14, 15),
            verbose=False,
        )
        print(f"[cal] frozen for missingness: {s_frozen}")

        for level in levels:
            df = evaluate_variant_level(variant, level, s_frozen)
            all_rows.append(df)

        # Save per-variant file (never overwritten by other variants)
        variant_path = RESULTS_DIR / f"missingness_{variant}.csv"
        pd.concat(all_rows[-len(levels):], ignore_index=True).to_csv(
            variant_path, index=False
        )
        print(f"[out] saved {variant_path}")

        # Save combined file (append + dedupe)
        df_combined = pd.concat(all_rows, ignore_index=True)
        df_combined = df_combined.drop_duplicates(
            subset=["variant", "target_missingness", "rep", "origin", "horizon"],
            keep="last",
        )
        df_combined.to_csv(combined_path, index=False)
        print(f"[out] combined saved: {combined_path}")

    df_all = pd.concat(all_rows, ignore_index=True).drop_duplicates(
        subset=["variant", "target_missingness", "rep", "origin", "horizon"],
        keep="last",
    )

    print("\n=== SUMMARY: RMSE and PICP by variant and missingness level ===")
    summary = df_all.groupby(["variant", "target_missingness"])[
        ["RMSE_orig", "PICP", "Width_orig", "CRPS_orig"]
    ].mean().round(3)
    print(summary)

    return df_all

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    if target == "all":
        run_all()
    else:
        run_all(target_variants=[target])