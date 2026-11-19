# tune.py — runs ONE (variant, rep, origin, horizon) cell and prints metrics.
import sys
from config import reset_seed, SEED
reset_seed(SEED)

from run_experiment import _load_and_prepare, _train_fn, evaluate_cell
from config import make_config, ORIGINS, HORIZONS, TEST_REPS
from data_io import load_truth_obs, prepare

variant = sys.argv[1] if len(sys.argv) > 1 else "slow_control"
rep     = int(sys.argv[2]) if len(sys.argv) > 2 else 16
origin  = int(sys.argv[3]) if len(sys.argv) > 3 else 80
horizon = int(sys.argv[4]) if len(sys.argv) > 4 else 1

cfg = make_config(variant, rep_for_params=rep)
_, obs = load_truth_obs(variant, rep)
df = prepare(obs, cfg)

s_frozen = {1: 1.0, 7: 1.0, 14: 1.0}   # identity calibration for now

row = evaluate_cell(df, cfg, origin, horizon, s_frozen, rep=rep, verbose=True)
print()
for k, v in row.items():
    print(f"  {k}: {v}")