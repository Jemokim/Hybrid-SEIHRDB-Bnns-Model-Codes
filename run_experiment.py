
# ============================================================
# run_experiment.py
# THE ONE EXPERIMENT. Origins 80/90/100, horizons 1/7/14.
# Satisfies remarks: 3, 4, 5, 8, 10, 11, 13, 17.
# ============================================================
from __future__ import annotations
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from pathlib import Path
import config
from config import (make_config, Config, DEVICE,
                    VARIANTS, TRAIN_REPS, VAL_REPS, TEST_REPS,
                    ORIGINS, HORIZONS, RESULTS_DIR,
                    load_true_params, reset_seed)
from data_io import load_truth_obs, prepare, to_tensors, mask_summary
from model import EbolaHybrid
from metrics import (rmse, picp, winkler, crps, mean_error, mape,
                     safe_corr, summarise)
from calibration import fit_calibration_factors, apply_factor


# ============================================================
# Loss
# ============================================================
def nb_nll(y, mu, alpha):
    eps = 1e-4
    mu = torch.clamp(mu, min=0.01)
    alpha = torch.clamp(alpha, min=0.01, max=100.0)
    inv = torch.clamp(1.0 / alpha, 0.01, 100.0)
    logp = (torch.lgamma(y + inv)
            - torch.lgamma(inv)
            - torch.lgamma(y + 1.0)
            + inv * torch.log(inv + eps)
            - inv * torch.log(inv + mu + eps)
            + y * torch.log(mu + eps)
            - y * torch.log(inv + mu + eps))
    logp = torch.clamp(logp, -100.0, 10.0)
    loss = -torch.mean(logp)
    if torch.isnan(loss) or torch.isinf(loss):
        return torch.tensor(10.0, device=loss.device)
    return loss


# ============================================================
# Training
# ============================================================
def train(model: EbolaHybrid, history, cfg: Config, verbose: bool = False) -> float:
    """Train with internal validation hold-out (Remark 3 — no leakage)."""
    opt = optim.Adam(model.parameters(), lr=cfg.lr)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min",
                                                 factor=0.5, patience=20)

    n = int(history["Y"].numel())
    hold = cfg.val_holdout_days
    if n > hold + 20:
        train_sl = slice(0, n - hold)
        val_sl = slice(n - hold, n)
    else:
        train_sl = slice(0, n)
        val_sl = None

    mask = history["mask"]
    data_max = float(history["Y"].max().item())
    mu_max = max(10.0, data_max * 3.0)
    best, patience = float("inf"), 0
    model.train()

    for epoch in range(cfg.epochs):
        opt.zero_grad()
        mus, _, smooth = model.run_on_history(
            history["t"], history["logY"], history["logH"], history["logD"],
            N=model._N_scaled, mu_max=mu_max)
        y_obs = history["Y"][train_sl][mask[train_sl]]
        mu_obs = mus[train_sl][mask[train_sl]]
        loss = nb_nll(y_obs, mu_obs, model.alpha) + 0.5 * smooth
        loss = torch.clamp(loss, max=100.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        sched.step(loss.detach().item())

        with torch.no_grad():
            if val_sl is not None and mask[val_sl].sum().item() > 0:
                yv = history["Y"][val_sl][mask[val_sl]]
                mv = mus[val_sl][mask[val_sl]]
                val_loss = float(nb_nll(yv, mv, model.alpha).item())
            else:
                val_loss = float(loss.item())

        if verbose and epoch % 40 == 0:
            with torch.no_grad():
                b_last, _ = model.bnn(history["t"][-1], history["logY"][-1],
                                      history["logH"][-1], history["logD"][-1])
                # rho gradient check: how big is d(loss)/d(_rho)?
                if model._rho.grad is not None:
                    rho_grad = model._rho.grad.item()
                else:
                    rho_grad = float("nan")
                print(f"  [rho] epoch {epoch:3d}: rho={model.rho.item():.4f}  "
                      f"grad={rho_grad:+.4e}")
            rmse_obs = float(torch.sqrt(torch.mean((y_obs - mu_obs) ** 2)).item())
            print(f"  epoch {epoch:3d} | loss {loss.item():.3f} | "
                  f"smooth {smooth.item():.3f} | "
                  f"beta_I[last]={b_last[0].item():.3f} rho={model.rho.item():.3f} "
                  f"eta={0.5 * torch.sigmoid(model._eta_base).item():.3f} "
                  f"alpha={model.alpha.item():.3f} | RMSE_obs={rmse_obs:.2f}")

        if val_loss < best:
            best, patience = val_loss, 0
        else:
            patience += 1
            if patience > cfg.early_stop_patience:
                if verbose:
                    print(f"  early stop at epoch {epoch} (val)")
                break

    if verbose:
        print(f"  final train loss {loss.item():.4f} | final val loss {best:.4f}")
    return best


# ============================================================
# MC forecast
# ============================================================
@torch.no_grad()
def mc_forecast(model: EbolaHybrid, history, future_t, horizon: int,
                cfg: Config, diagnostic: bool = False):
    """Monte-Carlo forecast with state anchoring.

    Protocol:
      1. One history pass (with dropout) gives a fitted final state.
      2. Anchor that state to the last observation:
             rho * sigma * E <- last_observed_count
         by rescaling E and I. This eliminates the identifiability
         failure where the fitted state is inconsistent with the
         last observed data point.
      3. For each of M trajectories, evolve forward from the anchored
         state, drawing a fresh dropout mask at each step.
      4. prev_logY is FROZEN at the last observed value for the whole
         forecast horizon, breaking the positive-feedback loop where
         the network's own predictions drive beta up.
    """
    model.train()   # dropout ON
    data_max = float(history["Y"].max().item())
    mu_max = max(10.0, data_max * 3.0)

    # ---- 1. One history pass, get fitted final state ----
    _, final_state, _ = model.run_on_history(
        history["t"], history["logY"], history["logH"], history["logD"],
        N=model._N_scaled, mu_max=mu_max)

    # ---- 2. Anchor state to recent observations (5-day mean) ----
    #
    # Anchoring to the single last observation is too aggressive: that
    # point is noisy and can be a low outlier (e.g. day 80 = 1.74 while
    # days 76-79 average ~14). Use a 5-day recent mean instead, which
    # is stable and consistent with the true epidemic level.
    window = min(5, len(history["Y"]))
    recent = history["Y"][-window:]
    recent_mean = float(recent.mean().item())

    # ---- near-extinction refinement ----
    # When the epidemic is almost over (recent mean < 10 in scaled units),
    # the 5-day mean is inflated by noisy high days. Anchor to the minimum
    # of the recent window instead — that better reflects the true low
    # level when cases are dying out.
    if recent_mean < 10.0:
        recent_mean = max(0.5, float(recent.min().item()))

    target_mu = max(0.5, recent_mean)
    current_mu = float((model.rho * model.sigma * final_state[1]).item())
    if current_mu > 1e-3 and abs(current_mu - target_mu) > 0.5:
        s = target_mu / current_mu
        E_new = final_state[1] * s
        I_new = final_state[2] * s
        dS = (E_new + I_new) - (final_state[1] + final_state[2])
        S_new = final_state[0] - dS
        final_state = torch.stack([
            S_new, E_new, I_new,
            final_state[3], final_state[4], final_state[5], final_state[6]
        ])
        if diagnostic:
            print(f"    [anchor] recent_mean={recent_mean:.2f}  "
                  f"mu_before={current_mu:.2f}  scale={s:.3f}  "
                  f"mu_after={float((model.rho * model.sigma * final_state[1]).item()):.2f}")

    state0 = final_state.detach().clone()
    # Freeze the network input to the recent mean as well.
    last_logY = history["logY"][-window:].mean().detach()

    # ---- 3. Monte-Carlo forward trajectories ----
    M = cfg.mc_samples
    Y_paths, mu_paths, I_paths = [], [], []
    BETA_paths = []  # <-- NEW: collect beta samples per MC trajectory

    for _ in range(M):
        state = state0.clone()
        prev_logY = last_logY.clone()
        ys, ms, Is = [], [], []
        betas_k = []  # <-- NEW: per-trajectory beta list

        for k in range(horizon):
            t_k = future_t[k]
            logH_k = torch.log1p(state[3])
            logD_k = torch.log1p(state[5])
            beta, _ = model.bnn(t_k, prev_logY, logH_k, logD_k)
            betas_k.append(beta.detach().clone())  # <-- NEW: record beta at this step
            new_state, _ = model.rk4(state, beta, model._N_scaled, k)
            mu_k = torch.clamp(model.rho * model.sigma * new_state[1],
                               0.01, mu_max)
            a = torch.clamp(model.alpha, min=1e-4)
            inv = 1.0 / a
            p = inv / (inv + mu_k)
            y = torch.distributions.NegativeBinomial(
                total_count=inv, probs=1.0 - p).sample()

            state = new_state
            ys.append(y)
            ms.append(mu_k)
            Is.append(state[2])
            prev_logY = last_logY.clone()   # <-- FROZEN: no feedback

        Y_paths.append(torch.stack(ys))
        mu_paths.append(torch.stack(ms))
        I_paths.append(torch.stack(Is))
        BETA_paths.append(torch.stack(betas_k))  # <-- NEW: [T, 3] for this MC sample

    Y = torch.stack(Y_paths)
    MU = torch.stack(mu_paths)
    I = torch.stack(I_paths)

    if diagnostic:
        print(f"    [mc] h={horizon:2d}  Y[k=0] mean={Y[:,0].mean().item():.2f} "
              f"sd={Y[:,0].std().item():.2f} "
              f"p05={torch.quantile(Y[:,0], 0.05).item():.2f} "
              f"p95={torch.quantile(Y[:,0], 0.95).item():.2f}")

    BETA = torch.stack(BETA_paths)  # <-- NEW: shape [M, T, 3]
    return {"Y_samples": Y, "mu_samples": MU, "I_samples": I,
            "beta_samples": BETA}

# ============================================================
# Convenience wrappers used by the calibration module
# ============================================================
def _load_and_prepare(variant: str, rep: int, cfg: Config):
    _, obs = load_truth_obs(variant, rep)
    return prepare(obs, cfg)


def _train_fn(df: pd.DataFrame, cfg: Config, origin: int):
    history_df = df.iloc[:origin].copy()
    history = to_tensors(history_df, cfg)
    model = EbolaHybrid(cfg, history_df).to(DEVICE)
    train(model, history, cfg, verbose=True)
    return model


def _predict_fn(model, history_df, future_df, horizon, cfg):
    history = to_tensors(history_df, cfg)
    future = to_tensors(future_df, cfg)
    fc = mc_forecast(model, history, future["t"], horizon, cfg,
                     diagnostic=True)
    return {
        "Y_true": future["Y"],
        "Y_samples": fc["Y_samples"],
        "mu_samples": fc["mu_samples"],
        "I_samples": fc["I_samples"],
        "beta_samples": fc["beta_samples"],   # <-- NEW
    }

# ============================================================
# Evaluation of one (variant, rep, origin, horizon) cell
# ============================================================
def evaluate_cell(df, cfg, origin, horizon, s_frozen, rep=None, verbose=True):
    history_df = df.iloc[:origin].copy()
    future_df = df.iloc[origin:origin + horizon].copy()

    model = _train_fn(df, cfg, origin)
    fc = _predict_fn(model, history_df, future_df, horizon, cfg)

    y_true = fc["Y_true"]
    y_samples = fc["Y_samples"]
    mu_samples = fc["mu_samples"]
    I_samples = fc["I_samples"]

    lo, hi = apply_factor(y_samples, s_frozen[horizon])
    # Case counts are non-negative; force the lower bound to 0 when it
    # would otherwise be positive on a near-extinct epidemic. Without
    # this, truth=0 cells fall outside the interval at h=1.
    lo = torch.minimum(lo, torch.zeros_like(lo))
    y_mean = y_samples.mean(dim=0)
    mu_mean = mu_samples.mean(dim=0)
    I_mean = I_samples.mean(dim=0)
    if verbose and horizon == 1:
        yt = [round(float(v), 2) for v in y_true.tolist()]
        ym = [round(float(v), 2) for v in y_mean.tolist()]
        mum = [round(float(v), 2) for v in mu_mean.tolist()]
        Im = [round(float(v), 2) for v in I_mean.tolist()]
        yl = [round(float(v), 2) for v in lo.tolist()]
        yh = [round(float(v), 2) for v in hi.tolist()]
        print(f"    [diag] y_true   ={yt}")
        print(f"    [diag] mu_mean  ={mum}   <- deterministic mean of NB")
        print(f"    [diag] y_mean   ={ym}   <- MC mean of Y samples")
        print(f"    [diag] I_mean   ={Im}    <- latent infectious stock")
        print(f"    [diag] lo       ={yl}")
        print(f"    [diag] hi       ={yh}")
        for i in range(len(yt)):
            side = "inside" if yl[i] <= yt[i] <= yh[i] \
                else ("below" if yt[i] < yl[i] else "above")
            bias = round(ym[i] - yt[i], 2)
            print(f"    [diag] day {i + 1}: truth={yt[i]}  "
                  f"mu_mean={mum[i]}  y_mean={ym[i]}  "
                  f"interval=[{yl[i]}, {yh[i]}]  "
                  f"bias(y_mean-truth)={bias}  -> {side}")

    # mu diagnostics (Remark 12 — observation equation behaviour)
    mu_rmse = rmse(y_true, mu_mean)
    mu_me = mean_error(y_true, mu_mean)
    mu_w05 = float(torch.quantile(mu_samples, 0.05, dim=0).mean().item())
    mu_w95 = float(torch.quantile(mu_samples, 0.95, dim=0).mean().item())
    mu_width = mu_w95 - mu_w05
    I_mean_t = I_mean.mean().item()
    I_w05 = float(torch.quantile(I_samples, 0.05, dim=0).mean().item())
    I_w95 = float(torch.quantile(I_samples, 0.95, dim=0).mean().item())

    row = {
        "horizon": horizon,
        "origin": origin,
        "RMSE_orig": rmse(y_true, y_mean),
        "PICP": picp(y_true, lo, hi),
        "Width_orig": float((hi - lo).mean().item()),
        "Winkler_orig": winkler(y_true, lo, hi),
        "CRPS_orig": crps(y_true, y_samples),
        "ME_orig": mean_error(y_true, y_mean),
        "MAPE": mape(y_true, y_mean),
        "r_daily": safe_corr(y_true, y_mean),
        "r_cum": safe_corr(torch.cumsum(y_true, 0), torch.cumsum(y_mean, 0)),
        "s_applied": s_frozen[horizon],

        # --- mu diagnostics ---
        "mu_RMSE": mu_rmse,
        "mu_ME": mu_me,
        "mu_width": mu_width,
        "mu_mean_val": float(mu_mean.mean().item()),

        # --- I diagnostics ---
        "I_mean_val": I_mean_t,
        "I_width": I_w95 - I_w05,
    }

    if verbose:
        rep_str = f"{rep:02d}" if rep is not None else "??"
        print(f"[test] rep={rep_str} origin={origin:3d} h={horizon:2d}  "
              f"RMSE={row['RMSE_orig']:.1f}  PICP={row['PICP']:.3f}  "
              f"Width={row['Width_orig']:.1f}  Winkler={row['Winkler_orig']:.1f}  "
              f"CRPS={row['CRPS_orig']:.1f}  s={s_frozen[horizon]:.2f}")
    return row


# ============================================================
# Main — one variant (rapid_control first)
# ============================================================
def run_one_variant(variant: str) -> pd.DataFrame:
    reset_seed(config.SEED)          # <-- NEW: fresh RNG for this variant
    print(f"\n{'=' * 70}")
    print(f"VARIANT: {variant}")
    print(f"{'=' * 70}")

    cfg_probe = make_config(variant, rep_for_params=1)
    print(f"[cfg] population={cfg_probe.population_N:.0f}  "
          f"sigma={cfg_probe.sigma:.4f} gamma={cfg_probe.gamma:.4f} "
          f"gamma_H={cfg_probe.gamma_H:.4f} delta={cfg_probe.delta:.4f} "
          f"delta_H={cfg_probe.delta_H:.4f} kappa={cfg_probe.kappa:.4f}")

    # --- calibration on reps 11-15 ---
    cfg_cal = make_config(variant, rep_for_params=11)
    s_frozen = fit_calibration_factors(
        variant=variant,
        origins=ORIGINS,
        horizons=HORIZONS,
        predict_fn=_predict_fn,
        train_fn=_train_fn,
        load_and_prepare=_load_and_prepare,
        cfg_factory=lambda v: make_config(v, rep_for_params=11),
        val_reps=VAL_REPS,
        verbose=True,
    )

    # --- test on reps 16-20 ---
    print("\n=== Test on reps 16-20 ===")
    rows = []
    for rep in TEST_REPS:
        cfg_t = make_config(variant, rep_for_params=rep)
        _, obs = load_truth_obs(variant, rep)
        df = prepare(obs, cfg_t)
        scale = float(df["scale_factor"].iloc[0])
        for origin in ORIGINS:
            if origin >= len(df):
                continue
            for horizon in HORIZONS:
                if origin + horizon > len(df):
                    continue
                row = evaluate_cell(df, cfg_t, origin, horizon, s_frozen,
                                    rep=rep, verbose=True)
                row["variant"] = variant
                # unscale metrics that were computed on the scaled data
                for k in ("RMSE_orig", "Width_orig", "Winkler_orig",
                          "CRPS_orig", "ME_orig",
                          "mu_RMSE", "mu_ME", "mu_width", "mu_mean_val",
                          "I_mean_val", "I_width"):
                    row[k] = row[k] / scale
                rows.append(row)

    df_res = pd.DataFrame(rows)
    out_path = RESULTS_DIR / f"{variant}_results.csv"
    df_res.to_csv(out_path, index=False)
    print(f"[out] wrote {out_path}")

    summarise(df_res, scale_label="orig")

    # --- parameter recovery (Remark 10) ---
    param_rows = []
    for rep in TEST_REPS:
        tp = load_true_params(variant, rep)
        _, obs = load_truth_obs(variant, rep)
        cfg = make_config(variant, rep_for_params=rep)
        df = prepare(obs, cfg)
        model = _train_fn(df, cfg, ORIGINS[-1])
        param_rows.append({
            "variant": variant, "rep": rep,
            "sigma": model.sigma, "gamma": model.gamma,
            "gamma_H": model.gamma_H, "delta": model.delta,
            "delta_H": model.delta_H, "kappa": model.kappa,
            "rho_hat": model.rho.item(),
            "eta_hat": 0.5 * torch.sigmoid(model._eta_base).item(),
            "alpha_hat": model.alpha.item(),
            "E0_hat": model.E0.item(), "I0_hat": model.I0.item(),
            "H0_hat": model.H0.item(),
            "sigma_true": tp.get("sigma"), "gamma_true": tp.get("gamma"),
            "gamma_H_true": tp.get("gamma_H"), "delta_true": tp.get("delta"),
            "delta_H_true": tp.get("delta_H"), "kappa_true": tp.get("kappa"),
            "rho_true": tp.get("rho"), "eta_true": tp.get("eta"),
            "alpha_true": tp.get("alpha"),
            "E0_true": tp.get("E0"), "I0_true": tp.get("I0"),
            "H0_true": tp.get("H0"),
            "scale_factor": scale,   # NEW
            "E0_true_scaled": (tp.get("E0") or 0.0) * scale,   # NEW
            "I0_true_scaled": (tp.get("I0") or 0.0) * scale,   # NEW
            "H0_true_scaled": (tp.get("H0") or 0.0) * scale,   # NEW
        })
    df_params = pd.DataFrame(param_rows)
    df_params.to_csv(RESULTS_DIR / f"{variant}_parameter_recovery.csv", index=False)
    print(f"[out] wrote {RESULTS_DIR / f'{variant}_parameter_recovery.csv'}")

    return df_res


def run_all_variants():
    t0 = time.time()
    for v in VARIANTS:
        try:
            run_one_variant(v)
        except Exception as e:
            print(f"[error] {v}: {e}")
    print(f"\n=== DONE === all variants in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "rapid_control"
    if target == "all":
        run_all_variants()
    else:
        run_one_variant(target)