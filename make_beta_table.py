from config import make_config, VARIANTS, ORIGINS, DEVICE
from data_io import to_tensors
from run_experiment import _train_fn, _load_and_prepare
import numpy as np, pandas as pd, torch
from pathlib import Path

REP = 16
ORIGIN = ORIGINS[0]
M = 50

rows = []
for variant in VARIANTS:
    cfg = make_config(variant, rep_for_params=REP)
    df = _load_and_prepare(variant, REP, cfg)
    model = _train_fn(df, cfg, ORIGIN)
    hist = to_tensors(df.iloc[:ORIGIN].copy(), cfg)

    model.train()
    with torch.no_grad():
        T = hist["t"].shape[0]
        B = torch.empty(M, T, 3, device=DEVICE)
        for m in range(M):
            for k in range(T):
                b, _ = model.bnn(hist["t"][k], hist["logY"][k],
                                 hist["logH"][k], hist["logD"][k])
                B[m, k] = b
    B = B.cpu().numpy()
    med = np.median(B, axis=0)   # [T, 3]
    for j, name in enumerate(["beta_I", "beta_H", "beta_D"]):
        rows.append({
            "variant": variant,
            "component": name,
            "min": float(med[:, j].min()),
            "max": float(med[:, j].max()),
            "mean": float(med[:, j].mean()),
        })

pd.DataFrame(rows).to_csv("results/beta_ranges.csv", index=False)
print(pd.DataFrame(rows).pivot(index="variant", columns="component",
                                values="mean").round(3))