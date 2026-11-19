# ============================================================
# run_all.py
# Runs every stage of the revision in the correct order.
# Usage:   python run_all.py            (rapid_control only)
#          python run_all.py all        (all 7 variants)
#          python run_all.py rapid_control --skip robustness
# ============================================================
from __future__ import annotations
import sys
import time
import traceback
from pathlib import Path

import run_experiment
import run_baselines
import run_robustness
import make_figures
from config import VARIANTS, RESULTS_DIR, FIGURES_DIR, reset_seed

def banner(title: str) -> None:
    print("\n" + "#" * 70)
    print(f"# {title}")
    print("#" * 70)


def run_stage(name: str, fn, *args, **kwargs) -> bool:
    banner(name)
    t0 = time.time()
    try:
        fn(*args, **kwargs)
        print(f"[{name}] OK in {(time.time() - t0) / 60:.1f} min")
        return True
    except Exception:
        print(f"[{name}] FAILED after {(time.time() - t0) / 60:.1f} min")
        traceback.print_exc()
        return False


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    skip = set()
    if "--skip" in sys.argv:
        skip = set(sys.argv[sys.argv.index("--skip") + 1].split(","))

    target = args[0] if args else "rapid_control"
    variants = VARIANTS if target == "all" else [target]

    print(f"Target: {target}")
    print(f"Skipping: {sorted(skip) if skip else 'nothing'}")

    overall_ok = True
    for v in variants:
        print(f"\n{'=' * 70}\nVARIANT: {v}\n{'=' * 70}")

        # Per-variant seed reset: makes `run_all.py all` and
        # `run_all.py <variant>` produce identical results for <variant>.
        reset_seed()

        if "experiment" not in skip:
            ok = run_stage(f"experiment / {v}",
                           run_experiment.run_one_variant, v)
            overall_ok &= ok

        if "baselines" not in skip:
            ok = run_stage(f"baselines / {v}",
                           run_baselines.evaluate, v)
            overall_ok &= ok

        if "robustness" not in skip:
            ok = run_stage(f"dropout / {v}",
                           run_robustness.dropout_sweep, v)
            overall_ok &= ok
            ok = run_stage(f"missingness / {v}",
                           run_robustness.missingness_sweep, v)
            overall_ok &= ok
            ok = run_stage(f"multi-seed / {v}",
                           run_robustness.multi_seed, v)
            overall_ok &= ok

        if "figures" not in skip:
            ok = run_stage(f"figures / {v}",
                           make_figures.make_all, v)
            overall_ok &= ok

    banner("ALL DONE")
    print(f"Results in: {RESULTS_DIR.resolve()}")
    print(f"Figures in: {FIGURES_DIR.resolve()}")
    print("Exit:", "success" if overall_ok else "with errors")


if __name__ == "__main__":
    main()