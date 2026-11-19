
# ============================================================
# make_figures.py
# Three-panel format ONLY. Origins 80/90/100, horizons 1/7/14.
# Satisfies remarks: 4, 5, 8.
# ============================================================
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from pathlib import Path

from config import (make_config, DEVICE, VARIANTS, TEST_REPS, ORIGINS,
                    HORIZONS, FIGURES_DIR, INTERVAL_LEVEL)
from data_io import load_truth_obs, prepare, to_tensors
from model import EbolaHybrid
from calibration import apply_factor, fit_calibration_factors
from run_experiment import (train, mc_forecast,
                            _train_fn, _predict_fn, _load_and_prepare)


def plot_three_panel(fc: dict, y_true: torch.Tensor, origin: int, horizon: int,
                     cal_s: float, save_path: Path):
    Y = fc["Y_samples"].cpu().numpy()
    MU = fc["mu_samples"].cpu().numpy()
    I = fc["I_samples"].cpu().numpy()
    y = y_true.cpu().numpy()

    y_mean = Y.mean(axis=0)
    y_lo0 = np.percentile(Y, 5, axis=0)
    y_hi0 = np.percentile(Y, 95, axis=0)
    y_lo = y_mean - cal_s * (y_mean - y_lo0)
    y_hi = y_mean + cal_s * (y_hi0 - y_mean)

    mu_mean = MU.mean(axis=0)
    mu_lo = np.percentile(MU, 5, axis=0)
    mu_hi = np.percentile(MU, 95, axis=0)

    I_mean = I.mean(axis=0)
    I_lo = np.percentile(I, 5, axis=0)
    I_hi = np.percentile(I, 95, axis=0)

    days = np.arange(1, horizon + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    level = int(INTERVAL_LEVEL * 100)

    ax1 = axes[0]
    ax1.plot(days, y, "ko-", linewidth=1.5, label="Observed $Y_t$")
    ax1.plot(days, y_mean, "r--", linewidth=1.8, label="Predicted median")
    ax1.fill_between(days, y_lo, y_hi, color="red", alpha=0.20,
                     label=f"{level}% CI")
    ax1.set_xlabel("Forecast day"); ax1.set_ylabel("Daily cases $Y_t$")
    ax1.set_title(f"Origin={origin}, Horizon={horizon}\n"
                  f"Daily reported cases (flow) — {level}% CI")
    ax1.grid(alpha=0.3); ax1.legend(loc="upper left", fontsize=9)
    ax1.text(0.02, 0.98, "$Y_t = \\rho \\sigma E(t)$",
             transform=ax1.transAxes, fontsize=9, va="top",
             bbox=dict(boxstyle="round", fc="white", ec="gray"))
    ax2 = axes[1]
    ax2.plot(days, I_mean, "g--", linewidth=1.8, label="$I(t)$ median")
    ax2.fill_between(days, I_lo, I_hi, color="green", alpha=0.20,
                     label=f"{level}% CI")
    ax2.set_xlabel("Forecast day"); ax2.set_ylabel("Infectious stock $I(t)$")
    ax2.set_title("Latent infectious stock (prevalence)\nNOT directly observed")
    ax2.grid(alpha=0.3); ax2.legend(loc="upper left", fontsize=9)

    ax3 = axes[2]
    ax3.plot(days, y, "ko-", linewidth=1.5, label="Observed $Y_t$")
    ax3.plot(days, mu_mean, "b--", linewidth=1.8, label="Predicted $\\mu_t$")
    ax3.fill_between(days, mu_lo, mu_hi, color="blue", alpha=0.20,
                     label=f"{level}% CI $\\mu_t$")
    ax3.set_xlabel("Forecast day")
    ax3.set_ylabel("Daily cases / incidence", color="blue")
    ax3.tick_params(axis="y", labelcolor="blue")
    ax3.grid(alpha=0.3)

    ax3b = ax3.twinx()
    ax3b.plot(days, I_mean, "g--", linewidth=1.8, label="Latent $I(t)$")
    ax3b.fill_between(days, I_lo, I_hi, color="green", alpha=0.15,
                      label=f"{level}% CI $I(t)$")
    ax3b.set_ylabel("Infectious stock $I(t)$", color="green")
    ax3b.tick_params(axis="y", labelcolor="green")

    ax3.set_title("Comparison: $Y_t$ (cases) vs $I(t)$ (stock)\n"
                  "$Y_t$ from S→E flow, not directly from $I(t)$")

    h1, l1 = ax3.get_legend_handles_labels()
    h2, l2 = ax3b.get_legend_handles_labels()
    ax3.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[fig] {save_path}")


def make_all(variant: str, reps=(16,)):
    # calibration factors on reps 11-15 (frozen, as in run_experiment)
    s_frozen = fit_calibration_factors(
        variant=variant, origins=ORIGINS, horizons=HORIZONS,
        predict_fn=_predict_fn, train_fn=_train_fn,
        load_and_prepare=_load_and_prepare,
        cfg_factory=lambda v: make_config(v, rep_for_params=11),
        val_reps=(11, 12, 13, 14, 15),
        verbose=False,
    )

    for rep in reps:
        cfg = make_config(variant, rep_for_params=rep)
        _, obs = load_truth_obs(variant, rep)
        df = prepare(obs, cfg)

        for origin in ORIGINS:
            if origin >= len(df):
                continue
            model = _train_fn(df, cfg, origin)
            for horizon in HORIZONS:
                if origin + horizon > len(df):
                    continue
                history_df = df.iloc[:origin].copy()
                future_df = df.iloc[origin:origin + horizon].copy()
                fc = _predict_fn(model, history_df, future_df, horizon, cfg)
                plot_three_panel(
                    fc, fc["Y_true"], origin, horizon,
                    cal_s=s_frozen[horizon],
                    save_path=FIGURES_DIR / f"{variant}_rep{rep}_origin{origin}_h{horizon}.png",
                )


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "rapid_control"
    variants = VARIANTS if target == "all" else [target]
    for v in variants:
        try:
            make_all(v)
        except Exception as e:
            print(f"[error] {v}: {e}")

