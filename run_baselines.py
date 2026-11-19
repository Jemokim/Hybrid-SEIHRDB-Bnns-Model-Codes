
# ============================================================
# run_baselines.py
# Zero, Persistence, Fixed SEIHRD, Deterministic NN, BNN.
# Satisfies remarks: 6, 7.
# ============================================================
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from config import (make_config, DEVICE, VARIANTS, TEST_REPS, ORIGINS,
                    HORIZONS, RESULTS_DIR)
from data_io import load_truth_obs, prepare, to_tensors
from model import EbolaHybrid, BayesianRateNN
from metrics import rmse, diebold_mariano
from run_experiment import train, mc_forecast, nb_nll


# ============================================================
# Zero and Persistence
# ============================================================
def zero_forecast(history_df, horizon):
    return torch.zeros(horizon, device=DEVICE)


def persistence_forecast(history_df, cfg, horizon):
    last = float(history_df[cfg.cases_col].iloc[-1])
    return torch.full((horizon,), last, device=DEVICE)


# ============================================================
# Fixed-rate SEIHRDB
# ============================================================
class FixedRateSEIHRD(nn.Module):
    def __init__(self, cfg, beta_I=0.2, beta_H=0.1, beta_D=0.05):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("beta", torch.tensor([beta_I, beta_H, beta_D],
                                                  device=DEVICE))
        self._rho = nn.Parameter(torch.tensor(0.0, device=DEVICE))
        self._eta = nn.Parameter(torch.tensor(0.0, device=DEVICE))
        self._alpha = nn.Parameter(torch.tensor(0.0, device=DEVICE))
        self._log_E0 = nn.Parameter(torch.log(torch.tensor(50.0, device=DEVICE)))
        self._log_I0 = nn.Parameter(torch.log(torch.tensor(50.0, device=DEVICE)))
        self._log_H0 = nn.Parameter(torch.log(torch.tensor(0.1, device=DEVICE)))

        self.sigma, self.gamma, self.gamma_H = cfg.sigma, cfg.gamma, cfg.gamma_H
        self.delta, self.delta_H, self.kappa = cfg.delta, cfg.delta_H, cfg.kappa
        self._N_scaled = float(cfg.population_N)

    @property
    def rho(self):   return torch.sigmoid(self._rho)
    @property
    def eta(self):   return 0.5 * torch.sigmoid(self._eta)
    @property
    def alpha(self): return torch.exp(self._alpha)
    @property
    def E0(self):    return torch.exp(self._log_E0).clamp(min=0.5, max=self._N_scaled * 0.05)
    @property
    def I0(self):    return torch.exp(self._log_I0).clamp(min=0.5, max=self._N_scaled * 0.05)
    @property
    def H0(self):    return torch.exp(self._log_H0).clamp(min=0.0, max=self._N_scaled * 0.05)

    def initial_state(self):
        E0, I0, H0 = self.E0, self.I0, self.H0
        S0 = self._N_scaled - (E0 + I0 + H0)
        z = torch.tensor(0.0, device=DEVICE)
        return torch.stack([S0, E0, I0, H0, z, z, z])

    def derivatives(self, state, N, t):
        S, E, I, H, R, D, B = state
        lam = (self.beta[0] * I + self.beta[1] * H + self.beta[2] * D) / N
        eta = self.eta
        dS = -lam * S
        dE = lam * S - self.sigma * E
        dI = self.sigma * E - (self.gamma + self.delta + eta) * I
        dH = eta * I - (self.gamma_H + self.delta_H) * H
        dR = self.gamma * I + self.gamma_H * H
        dD = self.delta * I + self.delta_H * H - self.kappa * D
        dB = self.kappa * D
        return torch.stack([dS, dE, dI, dH, dR, dD, dB]), lam

    def rk4(self, state, N, t):
        k1, lam1 = self.derivatives(state, N, t)
        k2, _ = self.derivatives(state + 0.5 * k1, N, t + 0.5)
        k3, _ = self.derivatives(state + 0.5 * k2, N, t + 0.5)
        k4, _ = self.derivatives(state + k3, N, t + 1.0)
        return torch.clamp(state + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0, min=0.0), lam1

    def run(self, t_span, state0=None, mu_max=1e5):  # <-- new arg
        if state0 is None:
            state0 = self.initial_state()
        state = state0.clone()
        mus = []
        for i in range(len(t_span)):
            state, lam = self.rk4(state, self._N_scaled, float(i))
            mus.append(torch.clamp(self.rho * self.sigma * state[1],
                                   0.01, mu_max))  # <-- use mu_max
        return torch.stack(mus)

    def fit(self, df, origin, cfg):
        self._N_scaled = float(df["population_scaled"].iloc[0])
        history_df = df.iloc[:origin].copy()
        h = to_tensors(history_df, cfg)
        opt = optim.Adam(self.parameters(), lr=1e-2)
        m = h["mask"]
        for epoch in range(cfg.epochs):
            opt.zero_grad()
            state0 = self.initial_state()  # recompute inside loop
            mu_max_fit = max(10.0, float(h["Y"].max().item()) * 3.0)
            mus = self.run(h["t"], state0, mu_max=mu_max_fit)
            loss = nb_nll(h["Y"][m], mus[m], self.alpha)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), cfg.grad_clip)
            opt.step()
        return self

    def forecast(self, horizon, state0=None, mu_max=1e5):  # <-- new arg
        if state0 is None:
            state0 = self.initial_state()
        dummy = torch.arange(horizon, device=DEVICE)
        return self.run(dummy, state0, mu_max=mu_max)


# ============================================================
# Deterministic NN (dropout = 0)
# ============================================================
class DeterministicNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.bnn = BayesianRateNN(cfg.hidden_dims, dropout_p=0.0,
                                  beta_scale=cfg.beta_scale)
        self._rho = nn.Parameter(torch.tensor(0.0, device=DEVICE))
        self._eta_base = nn.Parameter(torch.tensor(0.0, device=DEVICE))
        self._eta_amp = nn.Parameter(torch.tensor(0.0, device=DEVICE))
        self._alpha = nn.Parameter(torch.tensor(0.0, device=DEVICE))
        self._log_E0 = nn.Parameter(torch.log(torch.tensor(50.0, device=DEVICE)))
        self._log_I0 = nn.Parameter(torch.log(torch.tensor(50.0, device=DEVICE)))
        self._log_H0 = nn.Parameter(torch.log(torch.tensor(0.1, device=DEVICE)))
        self.sigma, self.gamma, self.gamma_H = cfg.sigma, cfg.gamma, cfg.gamma_H
        self.delta, self.delta_H, self.kappa = cfg.delta, cfg.delta_H, cfg.kappa
        self._N_scaled = float(cfg.population_N)

    @property
    def rho(self):   return torch.sigmoid(self._rho)
    @property
    def alpha(self): return torch.exp(self._alpha)
    @property
    def E0(self):    return torch.exp(self._log_E0).clamp(min=0.5, max=self._N_scaled * 0.05)
    @property
    def I0(self):    return torch.exp(self._log_I0).clamp(min=0.5, max=self._N_scaled * 0.05)
    @property
    def H0(self):    return torch.exp(self._log_H0).clamp(min=0.0, max=self._N_scaled * 0.05)

    def initial_state(self):
        E0, I0, H0 = self.E0, self.I0, self.H0
        S0 = self._N_scaled - (E0 + I0 + H0)
        z = torch.tensor(0.0, device=DEVICE)
        return torch.stack([S0, E0, I0, H0, z, z, z])

    def eta_t(self, t):
        base = 0.5 * torch.sigmoid(self._eta_base)
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=torch.float32, device=DEVICE)
        weekly = self._eta_amp * torch.sin(2 * torch.pi * t / 7)
        return torch.clamp(base + weekly, 0.01, 0.5)

    def derivatives(self, state, beta, N, t):
        S, E, I, H, R, D, B = state
        if beta.numel() != 3:
            beta = beta.reshape(-1)[:3]
        bI, bH, bD = beta
        lam = (bI * I + bH * H + bD * D) / N
        eta = self.eta_t(t)
        dS = -lam * S
        dE = lam * S - self.sigma * E
        dI = self.sigma * E - (self.gamma + self.delta + eta) * I
        dH = eta * I - (self.gamma_H + self.delta_H) * H
        dR = self.gamma * I + self.gamma_H * H
        dD = self.delta * I + self.delta_H * H - self.kappa * D
        dB = self.kappa * D
        return torch.stack([dS, dE, dI, dH, dR, dD, dB]), lam

    def rk4(self, state, beta, N, t, dt=1.0):
        k1, lam1 = self.derivatives(state, beta, N, t)
        k2, _ = self.derivatives(state + 0.5 * dt * k1, beta, N, t + 0.5 * dt)
        k3, _ = self.derivatives(state + 0.5 * dt * k2, beta, N, t + 0.5 * dt)
        k4, _ = self.derivatives(state + dt * k3, beta, N, t + dt)
        return torch.clamp(state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4), min=0.0), lam1

    def run_on_history(self, t, logY, logH, logD, state0=None, N=None, mu_max=1e5):
        if N is None:
            N = self._N_scaled
        if state0 is None:
            state0 = self.initial_state()
        state = state0.clone()
        mus, prev_beta = [], None
        for i in range(len(t)):
            if prev_beta is not None:
                beta, _ = self.bnn(t[i], logY[i], logH[i], logD[i], prev_beta)
            else:
                beta, _ = self.bnn(t[i], logY[i], logH[i], logD[i])
            state, lam = self.rk4(state, beta, N, i)
            # Observation equation: μ = ρ · σ · E (onset flow, matches BNN)
            mus.append(torch.clamp(self.rho * self.sigma * state[1], 0.01, mu_max))
            prev_beta = beta.detach()
        return torch.stack(mus), state

    def fit(self, df, origin, cfg):
        self._N_scaled = float(df["population_scaled"].iloc[0])
        history_df = df.iloc[:origin].copy()
        h = to_tensors(history_df, cfg)
        mu_max = max(10.0, float(h["Y"].max().item()) * 3.0)  # <-- NEW
        opt = optim.Adam(self.parameters(), lr=1e-2)
        m = h["mask"]
        for epoch in range(cfg.epochs):
            opt.zero_grad()
            state0 = self.initial_state()
            mus, _ = self.run_on_history(
                h["t"], h["logY"], h["logH"], h["logD"],
                state0=state0, N=self._N_scaled, mu_max=mu_max,  # <-- pass
            )
            loss = nb_nll(h["Y"][m], mus[m], self.alpha)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), cfg.grad_clip)
            opt.step()
        return self

    def forecast_deterministic(self, future_t, state0, horizon, mu_max=1e5):
        state = state0.clone()
        logY_prev = torch.log1p(torch.tensor(0.0, device=DEVICE))
        ys = []
        for k in range(horizon):
            logH = torch.log1p(state[3]);
            logD = torch.log1p(state[5])
            beta, _ = self.bnn(future_t[k], logY_prev, logH, logD)
            state, lam = self.rk4(state, beta, self._N_scaled, k)
            # Observation equation: μ = ρ · σ · E (onset flow, matches BNN)
            mu = torch.clamp(self.rho * self.sigma * state[1], 0.01, mu_max)
            ys.append(mu)
            logY_prev = torch.log1p(mu)
        return torch.stack(ys)


# ============================================================
# Evaluation harness
# ============================================================
def evaluate(variant: str):
    print(f"\n=== BASELINES: {variant} ===")
    rows = []
    for rep in TEST_REPS:
        cfg = make_config(variant, rep_for_params=rep)
        _, obs = load_truth_obs(variant, rep)
        df = prepare(obs, cfg)
        scale = float(df["scale_factor"].iloc[0])

        for origin in ORIGINS:
            if origin >= len(df):
                continue
            history_df = df.iloc[:origin].copy()
            for horizon in HORIZONS:
                if origin + horizon > len(df):
                    continue
                future_df = df.iloc[origin:origin + horizon].copy()
                y_true = torch.tensor(future_df[cfg.cases_col].values,
                                      dtype=torch.float32, device=DEVICE)

                preds = {}
                preds["Zero"] = zero_forecast(history_df, horizon)
                preds["Persistence"] = persistence_forecast(history_df, cfg, horizon)
                
                # Fixed
                fixed = FixedRateSEIHRD(cfg).to(DEVICE).fit(df, origin, cfg)
                hist = to_tensors(history_df, cfg)
                mu_max = max(10.0, float(hist["Y"].max().item()) * 3.0)
                preds["Fixed SEIHRD"] = fixed.forecast(horizon, mu_max=mu_max).detach()

                # Det NN
                det = DeterministicNN(cfg).to(DEVICE).fit(df, origin, cfg)
                fut = to_tensors(future_df, cfg)
                state0 = det.initial_state()
                hist = to_tensors(history_df, cfg)
                _, state_after = det.run_on_history(hist["t"], hist["logY"],
                                                    hist["logH"], hist["logD"],
                                                    state0, det._N_scaled,
                                                    mu_max=mu_max)
                preds["Det NN"] = det.forecast_deterministic(
                    fut["t"], state_after, horizon, mu_max=mu_max).detach()

                # BNN
                bnn = EbolaHybrid(cfg, history_df).to(DEVICE)
                hist = to_tensors(history_df, cfg)
                train(bnn, hist, cfg, verbose=False)
                fc = mc_forecast(bnn, hist, fut["t"], horizon, cfg)
                preds["BNN"] = fc["Y_samples"].mean(dim=0).detach()

                errs = {k: (y_true - v).cpu().numpy() for k, v in preds.items()}
                rmse_row = {k: rmse(y_true, v) / scale for k, v in preds.items()}

                row = {
                    "variant": variant, "rep": rep,
                    "origin": origin, "horizon": horizon,
                }
                for k in preds:
                    row[f"RMSE_{k}"] = rmse_row[k]

                # DM tests: BNN vs each
                for other in ("Zero", "Persistence", "Fixed SEIHRD", "Det NN"):
                    dm = diebold_mariano(errs["BNN"], errs[other])
                    row[f"DM_p_BNN_vs_{other}"] = dm["p_value"]
                    row[f"DM_stat_BNN_vs_{other}"] = dm["dm_stat"]

                rows.append(row)
                print(f"[baseline] {variant} rep={rep} origin={origin} h={horizon}  "
                      + "  ".join(f"{k}={rmse_row[k]:.2f}" for k in preds))

    df_res = pd.DataFrame(rows)
    out = RESULTS_DIR / f"{variant}_baselines.csv"
    df_res.to_csv(out, index=False)
    print(f"[out] {out}")
    return df_res


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "rapid_control"
    if target == "all":
        for v in VARIANTS:
            try:
                evaluate(v)
            except Exception as e:
                print(f"[error] {v}: {e}")
    else:
        evaluate(target)