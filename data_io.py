# ============================================================
# data_io.py
# Load, scale, tensorise. All masking is explicit and returned.
# Satisfies remarks: 2, 15.
# ============================================================
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict

from config import Config, BASE_DATA_DIR, DEVICE, ORIGINS


# ------------------------------------------------------------
# Raw loading
# ------------------------------------------------------------
def load_truth_obs(variant: str, rep: int):
    rep_dir = BASE_DATA_DIR / f"synthetic_{variant}" / f"rep_{rep:02d}"
    truth_p = rep_dir / "truth.csv"
    obs_p = rep_dir / "observations.csv"
    if not truth_p.exists() or not obs_p.exists():
        raise FileNotFoundError(f"Missing data files in {rep_dir}")
    return pd.read_csv(truth_p), pd.read_csv(obs_p)


# ------------------------------------------------------------
# Prepare for model
# ------------------------------------------------------------
def prepare(df_obs: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df_obs.rename(columns={
        "reported_cases": cfg.cases_col,
        "reported_hospital_admissions": cfg.hosp_col,
        "reported_deaths": cfg.deaths_col,
        "day": "t",
    }).copy()

    # missingness mask — explicit, returned for all downstream use
    if "is_missing_report" in df.columns:
        missing = df["is_missing_report"].fillna(False).astype(bool).values
        df["observed_mask"] = ~missing
    else:
        df["observed_mask"] = True

    for c in (cfg.cases_col, cfg.hosp_col, cfg.deaths_col):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).clip(lower=0.0)

    # Compute peak from history window only (min(ORIGINS) = 80) to prevent
    # future-observation leakage into the scaling constant.
    _history_end = min(ORIGINS)
    peak = float(df[cfg.cases_col].iloc[:_history_end].max())
    if cfg.scale_data and peak > cfg.target_max_cases:
        sf = cfg.target_max_cases / peak
        for c in (cfg.cases_col, cfg.hosp_col, cfg.deaths_col):
            df[c] = df[c] * sf
        df["scale_factor"] = sf
        df["population_scaled"] = cfg.population_N * sf
        print(f"[io] variant={cfg.variant} peak={peak:.0f}  "
              f"scale={sf:.5f}  N_scaled={df['population_scaled'].iloc[0]:.1f}")
    else:
        df["scale_factor"] = 1.0
        df["population_scaled"] = cfg.population_N
        print(f"[io] variant={cfg.variant} peak={peak:.0f}  scale=1.00000 (no scaling)")

    # time index and transforms
    df["t"] = np.arange(1, len(df) + 1, dtype=np.float32)
    df["t_norm"] = df["t"] / len(df)
    df["logY"] = np.log1p(df[cfg.cases_col]).astype(np.float32)
    df["logH"] = np.log1p(df[cfg.hosp_col]).astype(np.float32)
    df["logD"] = np.log1p(df[cfg.deaths_col]).astype(np.float32)

    for c in (cfg.cases_col, cfg.hosp_col, cfg.deaths_col, "t", "t_norm"):
        df[c] = df[c].astype(np.float32)

    return df


# ------------------------------------------------------------
# Tensor conversion
# ------------------------------------------------------------
def to_tensors(df: pd.DataFrame, cfg: Config) -> Dict[str, torch.Tensor]:
    def T(x, dtype=torch.float32):
        return torch.tensor(np.asarray(x), dtype=dtype, device=DEVICE)

    return {
        "t":      T(df["t_norm"].values),
        "Y":      T(df[cfg.cases_col].values),
        "H":      T(df[cfg.hosp_col].values),
        "D":      T(df[cfg.deaths_col].values),
        "logY":   T(df["logY"].values),
        "logH":   T(df["logH"].values),
        "logD":   T(df["logD"].values),
        "mask":   T(df["observed_mask"].values, dtype=torch.bool),
    }


# ------------------------------------------------------------
# Missingness summary (for the console contract)
# ------------------------------------------------------------
def mask_summary(history: Dict[str, torch.Tensor]) -> str:
    m = history["mask"]
    n = m.numel()
    n_obs = int(m.sum().item())
    return f"n_train={n}, n_obs={n_obs}, n_missing={n - n_obs}"
