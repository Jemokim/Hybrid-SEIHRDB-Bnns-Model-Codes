"""Figure 4.3: dropout sensitivity across variants."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = Path("results"); FIGURES_DIR = Path("figures"); FIGURES_DIR.mkdir(exist_ok=True)
VARIANTS = ["rapid_control","slow_control","early_control","high_transmission",
            "extreme_outbreak","moderate_control","multiple_waves"]

frames = []
for v in VARIANTS:
    f = RESULTS_DIR / f"{v}_dropout_sweep.csv"
    if f.exists():
        df = pd.read_csv(f); df["variant"] = v; frames.append(df)
if not frames:
    raise SystemExit("No *_dropout_sweep.csv found in results/.")

df = pd.concat(frames, ignore_index=True)
agg = df.groupby(["variant","dropout"]).agg(
    RMSE_mean=("RMSE","mean"), RMSE_std=("RMSE","std"),
    PICP_mean=("PICP","mean"), PICP_std=("PICP","std")).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
ax = axes[0]
for v, g in agg.groupby("variant"):
    ax.errorbar(g["dropout"], g["RMSE_mean"], yerr=g["RMSE_std"],
                marker="o", capsize=3, label=v, lw=1.4)
ax.set_xlabel("Dropout probability $p$"); ax.set_ylabel("RMSE (original scale)")
ax.set_title("(a) Point-forecast accuracy"); ax.grid(alpha=0.3)
ax.legend(fontsize=7)

ax = axes[1]
for v, g in agg.groupby("variant"):
    ax.errorbar(g["dropout"], g["PICP_mean"], yerr=g["PICP_std"],
                marker="s", capsize=3, label=v, lw=1.4)
ax.axhline(0.90, color="k", ls="--", lw=1, label="nominal 0.90")
ax.set_xlabel("Dropout probability $p$"); ax.set_ylabel("PICP")
ax.set_title("(b) Interval calibration"); ax.set_ylim(0, 1.05); ax.grid(alpha=0.3)

plt.tight_layout()
out = FIGURES_DIR / "fig_4_3_dropout_sensitivity.png"
plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()
print(f"[out] {out}")
