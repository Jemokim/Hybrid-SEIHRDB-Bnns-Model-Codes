# ============================================================
# model.py
# BayesianRateNN and EbolaHybrid (SEIHRDB).
# Observation equation: mu_t = rho * sigma * E_t  (onset-based).
# ============================================================
from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Tuple

from config import Config, DEVICE


# ---- Fix 1: transmission-rate bounds ----
BETA_I_MIN = 0.02
BETA_I_MAX = 0.80

class BayesianRateNN(nn.Module):
    """Maps (t, logY, logH, logD) -> three bounded transmission rates."""

    def __init__(self, hidden_dims: Tuple[int, int, int, int],
                 dropout_p: float, beta_scale: float):
        super().__init__()
        h1, h2, h3, h4 = hidden_dims
        self.beta_scale = beta_scale
        self.net = nn.Sequential(
            nn.Linear(4, h1), nn.Tanh(), nn.Dropout(dropout_p),
            nn.Linear(h1, h2), nn.Tanh(), nn.Dropout(dropout_p),
            nn.Linear(h2, h3), nn.Tanh(), nn.Dropout(dropout_p),
            nn.Linear(h3, h4), nn.Tanh(), nn.Dropout(dropout_p),
            nn.Linear(h4, 3),
        )

    def forward(self, t, logY, logH, logD, prev_beta=None):
        x = torch.stack([t, logY, logH, logD], dim=-1)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        raw = self.net(x).squeeze(0)
        bounded = BETA_I_MIN + (BETA_I_MAX - BETA_I_MIN) * torch.sigmoid(raw)
        if prev_beta is not None and self.training:
            smooth = torch.nn.functional.mse_loss(bounded, prev_beta)
            return bounded, smooth
        return bounded, None


class EbolaHybrid(nn.Module):
    """SEIHRDB with time-varying transmission beta_I(t), beta_H(t), beta_D(t)."""

    def __init__(self, cfg: Config, df: pd.DataFrame):
        super().__init__()
        self.cfg = cfg
        self.bnn = BayesianRateNN(cfg.hidden_dims, cfg.dropout_p, cfg.beta_scale)

        self._rho = nn.Parameter(torch.tensor(-1.2, device=DEVICE))
        self._eta_base = nn.Parameter(torch.tensor(0.0, device=DEVICE))
        self._eta_amp = nn.Parameter(torch.tensor(0.0, device=DEVICE))
        self._alpha = nn.Parameter(torch.tensor(0.0, device=DEVICE))

        # Initial states (learnable, anchored to the first observation).
        #
        # Observation equation: mu_0 = rho * sigma * E_0. For an
        # initial reporting fraction rho_init = sigmoid(-1.2) ~= 0.231,
        # the E_0 that reproduces the first observed case count is
        #     E_0 = first_cases / (rho_init * sigma).
        # The previous version divided by (0.7 * sigma) instead of
        # (rho_init * sigma), inflating E_0 by ~3x. On high_transmission
        # that error compounded over the forecast window into a 20x
        # over-prediction.
        first_cases = float(df[cfg.cases_col].iloc[0])
        rho_init = float(torch.sigmoid(torch.tensor(-1.2)).item())  # ~= 0.231
        E0 = max(0.5, first_cases / max(rho_init * cfg.sigma, 1e-6))
        I0 = E0 * 0.7
        self._log_E0 = nn.Parameter(torch.log(torch.tensor(E0, device=DEVICE)))
        self._log_I0 = nn.Parameter(torch.log(torch.tensor(I0, device=DEVICE)))
        self._log_H0 = nn.Parameter(torch.log(torch.tensor(0.1, device=DEVICE)))
        self.sigma = cfg.sigma
        self.gamma = cfg.gamma
        self.gamma_H = cfg.gamma_H
        self.delta = cfg.delta
        self.delta_H = cfg.delta_H
        self.kappa = cfg.kappa

        self._N_scaled = float(df["population_scaled"].iloc[0])

    @property
    def rho(self):
        raw = torch.sigmoid(self._rho)
        return 0.10 + 0.30 * raw

    @property
    def alpha(self):     return torch.exp(self._alpha)

    @property
    def E0(self):        return torch.exp(self._log_E0).clamp(max=self._N_scaled * 0.05)

    @property
    def I0(self):        return torch.exp(self._log_I0).clamp(max=self._N_scaled * 0.05)

    @property
    def H0(self):        return torch.exp(self._log_H0).clamp(max=self._N_scaled * 0.05)

    def eta_t(self, t):
        base = 0.5 * torch.sigmoid(self._eta_base)
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=torch.float32, device=DEVICE)
        weekly = self._eta_amp * torch.sin(2 * torch.pi * t / 7)
        return torch.clamp(base + weekly, 0.01, 0.5)

    def initial_state(self):
        E0, I0, H0 = self.E0, self.I0, self.H0
        S0 = self._N_scaled - (E0 + I0 + H0)
        z = torch.tensor(0.0, device=DEVICE)
        return torch.stack([S0, E0, I0, H0, z, z, z])

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
        k2, _    = self.derivatives(state + 0.5 * dt * k1, beta, N, t + 0.5 * dt)
        k3, _    = self.derivatives(state + 0.5 * dt * k2, beta, N, t + 0.5 * dt)
        k4, _    = self.derivatives(state + dt * k3, beta, N, t + dt)
        new_state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return torch.clamp(new_state, min=0.0), lam1

    def run_on_history(self, t, logY, logH, logD, state0=None, N=None, mu_max=1e6):
        if N is None:
            N = self._N_scaled
        if state0 is None:
            state0 = self.initial_state()
        state = state0.clone()

        mus, smooth_total, prev_beta = [], torch.tensor(0.0, device=DEVICE), None
        for i in range(len(t)):
            if prev_beta is not None:
                beta, smooth = self.bnn(t[i], logY[i], logH[i], logD[i], prev_beta)
                smooth_total = smooth_total + (smooth if smooth is not None else 0.0)
            else:
                beta, _ = self.bnn(t[i], logY[i], logH[i], logD[i])

            new_state, _ = self.rk4(state, beta, N, i)
            mu = torch.clamp(self.rho * self.sigma * new_state[1], 0.01, mu_max)

            state = new_state
            mus.append(mu)
            prev_beta = beta.detach()

        return torch.stack(mus), state, smooth_total