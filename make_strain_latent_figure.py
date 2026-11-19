"""
Generates a 2x2 grid of I(t)-vs-observed-cases panels, one per Ebola strain.
Saves to figures/strain_latent_comparison.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path

from config import reset_seed, SEED, make_config, DEVICE
from data_io import load_truth_obs, prepare, to_tensors
from model import EbolaHybrid
from run_experiment import train, mc_forecast

reset_seed(SEED)

# One variant per strain
REPRESENTATIVES = {
    "early_control":     "Bundibugyo (CFR ≈ 30%)",
    "moderate_control":  "Sudan (CFR ≈ 50%)",
    "slow_control":      "Zaire typical (CFR ≈ 60%)",
    "high_transmission": "Zaire severe (CFR ≈ 70%)",
}

REP = 16
ORIGIN = 80
HORIZON = 7

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle(
    f"Latent infectious stock $I(t)$ vs observed cases across Ebola strains "
    f"(rep {REP}, origin {ORIGIN}, horizon {HORIZON})",
    fontsize=14, fontweight="bold",
)

for ax, (variant, label) in zip(axes.flat, REPRESENTATIVES.items()):
    cfg = make_config(variant, rep_for_params=REP)
    _, obs = load_truth_obs(variant, REP)
    df_full = prepare(obs, cfg)

    history_df = df_full.iloc[:ORIGIN].copy()
    future_df = df_full.iloc[ORIGIN:ORIGIN + HORIZON].copy()

    history = to_tensors(history_df, cfg)
    model = EbolaHybrid(cfg, history_df).to(DEVICE)
    train(model, history, cfg, verbose=False)

    future = to_tensors(future_df, cfg)
    fc = mc_forecast(model, history, future["t"], HORIZON, cfg)

    I_samples = fc["I_samples"].cpu().numpy()
    I_median = np.median(I_samples, axis=0)
    I_lo = np.quantile(I_samples, 0.05, axis=0)
    I_hi = np.quantile(I_samples, 0.95, axis=0)

    y_obs = future["Y"].cpu().numpy()
    days = np.arange(1, HORIZON + 1)

    ax.plot(days, y_obs, "o-", color="black", lw=1.6,
            label="Observed $Y_t$")
    ax.plot(days, I_median, "--", color="green", lw=1.6,
            label="Latent $I(t)$ median")
    ax.fill_between(days, I_lo, I_hi, color="green", alpha=0.15,
                    label="90% CI $I(t)$")
    ax.set_xlabel("Forecast day")
    ax.set_ylabel("Cases / infectious stock")
    ax.set_title(label)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

plt.tight_layout()
out = Path("figures/strain_latent_comparison.png")
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"[out] {out}")