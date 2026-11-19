"""Figure 4.4: lead-time RMSE with quantile bands."""
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt

RESULTS_DIR = Path("results"); FIGURES_DIR = Path("figures"); FIGURES_DIR.mkdir(exist_ok=True)
VARIANTS = ["rapid_control","slow_control","early_control","high_transmission",
            "extreme_outbreak","moderate_control","multiple_waves"]

frames = []
for v in VARIANTS:
    f = RESULTS_DIR / f"{v}_multi_seed.csv"
    if f.exists():
        frames.append(pd.read_csv(f))
if not frames:
    raise SystemExit("No *_multi_seed.csv found in results/.")

df = pd.concat(frames, ignore_index=True)
peaks = {v: df[df["variant"]==v]["RMSE"].quantile(0.95) for v in df["variant"].unique()}
df["RMSE_norm"] = df.apply(lambda r: r["RMSE"]/peaks[r["variant"]], axis=1)
h = sorted(df["horizon"].unique())

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
ax = axes[0]
med, q25, q75 = [], [], []
for hh in h:
    g = df[df["horizon"]==hh]["RMSE_norm"]
    med.append(g.median()); q25.append(g.quantile(0.25)); q75.append(g.quantile(0.75))
ax.plot(h, med, "-o", color="C0", label="median")
ax.fill_between(h, q25, q75, color="C0", alpha=0.25, label="25-75% band")
ax.set_xlabel("Forecast horizon (days)"); ax.set_ylabel("RMSE / variant peak")
ax.set_title("(a) Normalized lead-time error, all variants")
ax.set_xticks(h); ax.grid(alpha=0.3); ax.legend(fontsize=9)

ax = axes[1]
med, q25, q75 = [], [], []
for hh in h:
    g = df[df["horizon"]==hh]["PICP"]
    med.append(g.median()); q25.append(g.quantile(0.25)); q75.append(g.quantile(0.75))
ax.plot(h, med, "-s", color="C1", label="median PICP")
ax.fill_between(h, q25, q75, color="C1", alpha=0.25, label="25-75% band")
ax.axhline(0.90, color="k", ls="--", lw=1, label="nominal 0.90")
ax.set_xlabel("Forecast horizon (days)"); ax.set_ylabel("PICP")
ax.set_title("(b) Coverage vs horizon"); ax.set_ylim(0, 1.05)
ax.set_xticks(h); ax.grid(alpha=0.3); ax.legend(fontsize=9)

plt.tight_layout()
out = FIGURES_DIR / "fig_4_4_leadtime_error.png"
plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()
print(f"[out] {out}")
