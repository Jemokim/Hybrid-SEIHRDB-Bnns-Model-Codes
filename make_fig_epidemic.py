"""Figure 4.5: epidemic curve with forecast origins."""
from pathlib import Path
import numpy as np, matplotlib.pyplot as plt
from config import make_config, ORIGINS
from data_io import load_truth_obs, prepare

FIGURES_DIR = Path("figures"); FIGURES_DIR.mkdir(exist_ok=True)
VARIANT = "rapid_control"; REP = 16

cfg = make_config(VARIANT, rep_for_params=REP)
_, obs = load_truth_obs(VARIANT, REP)
df = prepare(obs, cfg)

y = df[cfg.cases_col].fillna(0.0).values
mask = df["observed_mask"].values.astype(bool) if "observed_mask" in df \
       else np.ones_like(y, dtype=bool)
t = np.arange(len(y))

fig, ax = plt.subplots(figsize=(9, 4.2))
ax.plot(t, y, "-", color="black", lw=1.2, label="Daily cases")
ax.scatter(t[mask], y[mask], s=12, color="black", zorder=3)
ax.scatter(t[~mask], y[~mask], s=14, color="red", marker="x",
           zorder=4, label="Missing (masked)")
for o in ORIGINS:
    ax.axvline(o, color="C0", ls="--", lw=1, alpha=0.8)
    ax.text(o+0.5, ax.get_ylim()[1]*0.92, f"origin {o}",
            color="C0", fontsize=8, rotation=90, va="top")
ax.axvspan(0, ORIGINS[-1], alpha=0.05, color="green")
ax.text(2, ax.get_ylim()[1]*0.97, "history window",
        color="green", fontsize=9, va="top")
ax.set_xlabel("Day"); ax.set_ylabel("Reported cases")
ax.set_title(f"Epidemic curve with forecast origins and missing days "
             f"({VARIANT}, rep {REP})")
ax.grid(alpha=0.3); ax.legend(loc="best", fontsize=9)
plt.tight_layout()
out = FIGURES_DIR / "fig_4_5_epidemic_split.png"
plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()
print(f"[out] {out}")
