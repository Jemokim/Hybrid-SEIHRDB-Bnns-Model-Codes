"""
Figure 4.1: inferred transmission rates beta(t).
(a) over the training window (dynamic, observed data driving the network)
(b) over the 14-day forecast horizon (frozen input by design)
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch

from config import make_config, ORIGINS, HORIZONS, DEVICE
from data_io import to_tensors
from run_experiment import _train_fn, _load_and_prepare, mc_forecast

FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

VARIANT = "rapid_control"
REP     = 16
ORIGIN  = ORIGINS[0]     # 80
HORIZON = max(HORIZONS)  # 14

# ---------- train one model ----------
cfg   = make_config(VARIANT, rep_for_params=REP)
df    = _load_and_prepare(VARIANT, REP, cfg)
model = _train_fn(df, cfg, ORIGIN)

hist = to_tensors(df.iloc[:ORIGIN].copy(), cfg)
fut  = to_tensors(df.iloc[ORIGIN:ORIGIN + HORIZON].copy(), cfg)

# ---------- (a) beta over the training window ----------
model.train()   # keep dropout ON for consistency with mc_forecast
with torch.no_grad():
    M_hist = 50
    T_hist = hist["t"].shape[0]
    B_hist = torch.empty(M_hist, T_hist, 3, device=DEVICE)
    for m in range(M_hist):
        for k in range(T_hist):
            tk     = hist["t"][k]
            logY_k = hist["logY"][k]
            logH_k = hist["logH"][k]
            logD_k = hist["logD"][k]
            b, _ = model.bnn(tk, logY_k, logH_k, logD_k)
            B_hist[m, k] = b
B_hist = B_hist.cpu().numpy()

# ---------- (b) beta over the forecast horizon ----------
fc = mc_forecast(model, hist, fut["t"], HORIZON, cfg)
B_fc = fc["beta_samples"].cpu().numpy()   # [M, T, 3]

# ---------- plotting ----------
t_hist = np.arange(1, T_hist + 1)
t_fc   = np.arange(1, HORIZON + 1)

labels = [
    r"$\beta_I(t)$  community",
    r"$\beta_H(t)$  hospital",
    r"$\beta_D(t)$  funerary",
]
colors = ["C0", "C3", "C2"]

fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))

for j, ax in enumerate(axes[0]):
    med = np.median(B_hist[:, :, j], axis=0)
    lo  = np.quantile(B_hist[:, :, j], 0.05, axis=0)
    hi  = np.quantile(B_hist[:, :, j], 0.95, axis=0)
    ax.plot(t_hist, med, "-", color=colors[j], lw=1.5)
    ax.fill_between(t_hist, lo, hi, color=colors[j], alpha=0.20)
    ax.set_title(labels[j] + " — training window")
    ax.set_xlabel("Day")
    ax.set_ylabel("Transmission rate")
    ax.grid(alpha=0.3)

for j, ax in enumerate(axes[1]):
    med = np.median(B_fc[:, :, j], axis=0)
    lo  = np.quantile(B_fc[:, :, j], 0.05, axis=0)
    hi  = np.quantile(B_fc[:, :, j], 0.95, axis=0)
    ax.plot(t_fc, med, "-o", color=colors[j], lw=1.8, ms=5)
    ax.fill_between(t_fc, lo, hi, color=colors[j], alpha=0.20)
    ax.set_title(labels[j] + " — forecast horizon")
    ax.set_xlabel("Forecast day")
    ax.set_ylabel("Transmission rate")
    ax.set_xticks(t_fc)
    ax.grid(alpha=0.3)

fig.suptitle(
    f"Inferred transmission rates, {VARIANT.replace('_',' ').title()} "
    f"(rep {REP}, origin {ORIGIN} d)",
    fontsize=12,
)
plt.tight_layout(rect=[0, 0, 1, 0.96])

out = FIGURES_DIR / "fig_4_1_inferred_betas.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.close()
print(f"[out] {out}")
print(f"[hist] beta_I range over training: "
      f"{np.median(B_hist[:, :, 0], axis=0).min():.3f} – "
      f"{np.median(B_hist[:, :, 0], axis=0).max():.3f}")
print(f"[hist] beta_H range over training: "
      f"{np.median(B_hist[:, :, 1], axis=0).min():.3f} – "
      f"{np.median(B_hist[:, :, 1], axis=0).max():.3f}")
print(f"[hist] beta_D range over training: "
      f"{np.median(B_hist[:, :, 2], axis=0).min():.3f} – "
      f"{np.median(B_hist[:, :, 2], axis=0).max():.3f}")