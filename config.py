# ============================================================
# config.py
# Single source of truth for the BNN-SEIHRD revision.
# Satisfies supervisor remarks: 5, 8, 11, 16.
# ============================================================
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple, List
import json
import torch

# ---------- Reproducibility ----------
SEED = 42


def reset_seed(seed: int = SEED) -> None:
    """Reset every RNG used in the project to a known state.

    Call this at the start of every variant so that running `all` and
    running a single variant produce identical results for that variant.
    """
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

reset_seed(SEED)
# ---------- Paths ----------
BASE_DATA_DIR: Path = Path("synthetic_data_strains")
RESULTS_DIR: Path = Path("results")
FIGURES_DIR: Path = Path("figures")
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# ---------- The one experiment (Remark 5) ----------
VARIANTS: List[str] = [
    "rapid_control",
    "slow_control",
    "high_transmission",
    "extreme_outbreak",
    "moderate_control",
    "early_control",
    "multiple_waves",
]
TRAIN_REPS: Tuple[int, ...] = tuple(range(1, 11))     # 1..10
VAL_REPS:   Tuple[int, ...] = (11, 12, 13, 14, 15)
TEST_REPS:  Tuple[int, ...] = (16, 17, 18, 19, 20)
ORIGINS:    Tuple[int, ...] = (80, 90, 100)
HORIZONS:   Tuple[int, ...] = (1, 7, 14)

# ---------- Interval level (Remark 4) ----------
INTERVAL_LEVEL: float = 0.90
LOWER_Q: float = (1.0 - INTERVAL_LEVEL) / 2.0   # 0.05
UPPER_Q: float = 1.0 - LOWER_Q                   # 0.95
ALPHA_CI: float = 1.0 - INTERVAL_LEVEL           # 0.10 for Winkler

# ---------- Data scaling ----------
TARGET_MAX_CASES: float = 100.0
POPULATION_N_DEFAULT: float = 1_000_000.0

# ---------- Variant populations ----------
VARIANT_POPULATION: Dict[str, int] = {
    "rapid_control": 1_000_000,
    "slow_control": 250_000,
    "high_transmission": 200_000,
    "extreme_outbreak": 6_300_000,
    "moderate_control": 2_500_000,
    "early_control": 695_000,
    "multiple_waves": 1_000_000,
}

# ---------- Device ----------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Mechanistic parameters — MUST come from the generator (Remark 11)
# ============================================================
_MECH_KEYS = ("sigma", "gamma", "gamma_H", "delta", "delta_H", "kappa")


def load_true_params(variant: str, rep: int) -> Dict[str, float]:
    """Return the generator's mechanistic parameters for one replicate.

    Raises FileNotFoundError or KeyError with a clear message rather than
    falling back to defaults. This is deliberate: silently defaulting is
    exactly the mistake Remark 11 asks us to remove.
    """
    path = BASE_DATA_DIR / f"synthetic_{variant}" / f"rep_{rep:02d}" / "true_params.json"
    if not path.exists():
        raise FileNotFoundError(
            f"true_params.json missing for {variant} rep {rep} at {path}. "
            f"Regenerate the synthetic data or add the file before rerunning."
        )
    with open(path) as f:
        tp = json.load(f)
    missing = [k for k in _MECH_KEYS if k not in tp]
    if missing:
        raise KeyError(
            f"true_params.json for {variant} rep {rep} is missing keys: {missing}. "
            f"Expected at least {list(_MECH_KEYS)}."
        )
    return tp


# ============================================================
# Model configuration dataclass
# ============================================================
@dataclass
class Config:
    # Data
    variant: str = "rapid_control"
    population_N: float = POPULATION_N_DEFAULT
    cases_col: str = "cases"
    hosp_col: str = "hospitalizations"
    deaths_col: str = "deaths"
    scale_data: bool = True
    target_max_cases: float = TARGET_MAX_CASES

    # Network
    hidden_dims: Tuple[int, int, int, int] = (32, 64, 32, 16)
    dropout_p: float = 0.10
    lr: float = 1e-3
    epochs: int = 600
    mc_samples: int = 50
    grad_clip: float = 0.5
    early_stop_patience: int = 40
    val_holdout_days: int = 10

    # Mechanistic (loaded from true_params.json — do NOT hardcode)
    sigma: float = 1 / 8.5
    gamma: float = 1 / 13.0
    gamma_H: float = 1 / 10.5
    delta: float = 1 / 9.3
    delta_H: float = 1 / 4.6
    kappa: float = 1 / 5.0

    # β scaling in the BNN (no hard clamp — see model.py)
    beta_scale: float = 0.5

    # Intervals
    interval_lower_q: float = LOWER_Q
    interval_upper_q: float = UPPER_Q
    target_picp: float = INTERVAL_LEVEL

    def sync_mechanistic(self, variant: str, rep: int) -> "Config":
        """Overwrite sigma/gamma/... from the generator's true_params.json."""
        tp = load_true_params(variant, rep)
        for k in _MECH_KEYS:
            setattr(self, k, float(tp[k]))
        self.variant = variant
        return self


def make_config(variant: str, rep_for_params: int = 1) -> Config:
    """Convenience constructor used by every entry point."""
    cfg = Config()
    cfg.population_N = float(VARIANT_POPULATION.get(variant, POPULATION_N_DEFAULT))
    cfg.sync_mechanistic(variant, rep_for_params)
    return cfg