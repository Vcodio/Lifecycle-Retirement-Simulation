# ============================================================
# Version 3.0: Maximum Likelihood Estimation - Bates Model
#
# This script uses maximum likelihood estimation to fit the 
# Bates model to financial data from a CSV file sourced from, 
# Maximum Likelihood Estimation - Bates Model# testfolio.
#
# Changes from Version 2.1:
# - Overhauled to fit to Bates model instead of Merton Jump diffusion
# - Several major performance upgrades and bug fixes
# - Several bug improvements and performance upgrades
#
#
# Author: Vcodio
# ============================================================

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from rich.console import Console
from rich.table import Table
from numpy.random import default_rng
import math
import sys
import os

console = Console()

# -----------------------
# User configuration (edit if needed)
# -----------------------
csv_filename = 'Monte Carlo Asset Analyzer.csv'   # file with daily price levels
days_in_year = 252
DT = 1.0 / days_in_year

ASSET_NAMES = ['VTISIM','SPYSIM?L=2','VXUSSIM','DFSVX','DISVX','ZROZSIM']
ACCUMULATION_WEIGHTS = np.array([0.3, 0.2, 0.15, 0.10, 0.05, 0.2])
WITHDRAWAL_WEIGHTS   = np.array([0.6, 0.0, 0.2, 0.0, 0.0, 0.2])

NUM_OPTIMIZER_RESTARTS = 6
MAXITER = 1500
DENSITY_FLOOR = 1e-12

# Toggle: fit jumps (Bates) or Heston-only
WITH_JUMPS = True

# Parameter bounds (ANNUAL units) - adjust if you have domain knowledge
BOUNDS = [
    (-0.5, 0.5),     # mu (annual drift)
    (1e-6, 50.0),    # kappa (1/yr)
    (1e-12, 2.0),    # theta (annual variance)
    (1e-6, 5.0),     # nu (vol-of-vol annual)
    (-0.999, 0.999), # rho
    (1e-12, 2.0),    # v0 (annual variance)
    (0.0, 500.0),    # lambda (jumps per year)
    (-5.0, 5.0),     # mu_J (jump mean, log-return)
    (1e-12, 5.0),    # sigma_J (jump std)
]

# Index mapping for readability
IDX = {
    'mu': 0, 'kappa': 1, 'theta': 2, 'nu': 3, 'rho': 4, 'v0': 5,
    'lam': 6, 'mu_J': 7, 'sigma_J': 8
}

# -----------------------
# Helpers: normal logpdf
# -----------------------
def norm_logpdf(x, mean, sd):
    if sd <= 0 or not math.isfinite(sd):
        return -1e8
    return -0.5*math.log(2*math.pi) - math.log(sd) - 0.5*((x-mean)/sd)**2

# -----------------------
# Step log-likelihood (QMLE)
# - params are in ANNUAL units
# -----------------------
def step_loglik(r_t, v_prev, params):
    # unpack
    mu = params[IDX['mu']]
    kappa = params[IDX['kappa']]
    theta = params[IDX['theta']]
    nu = params[IDX['nu']]
    rho = params[IDX['rho']]
    v0 = params[IDX['v0']]
    lam = params[IDX['lam']]
    mu_J = params[IDX['mu_J']]
    sigma_J = params[IDX['sigma_J']]

    dt = DT
    # Gaussian approx for jump compound Poisson
    mJ = lam*dt*mu_J
    sJ2 = lam*dt*(sigma_J**2 + mu_J**2)

    # expected v_t given v_prev under CIR mean reversion (exact expectation)
    Ev_t = theta + (v_prev - theta)*math.exp(-kappa*dt)
    phi = (1 - math.exp(-kappa*dt))
    # trapezoid proxy for average variance across interval (more stable)
    v_bar = (1 - 0.5*phi)*v_prev + 0.5*phi*Ev_t

    mean_r = mu*dt + mJ
    var_r = v_bar*dt + sJ2

    if not math.isfinite(var_r) or var_r <= 0:
        return -1e8

    sd = math.sqrt(var_r)
    return norm_logpdf(r_t, mean_r, sd)

# -----------------------
# Full negative log-likelihood (params = vector of annual params)
# -----------------------
def neg_log_likelihood(params, returns, with_jumps=True):
    # params: length 9 vector (mu,kappa,theta,nu,rho,v0,lam,muJ,sigJ)
    # enforce domain quickly
    if np.any(~np.isfinite(params)):
        return 1e12
    mu = params[IDX['mu']]
    kappa = params[IDX['kappa']]
    theta = params[IDX['theta']]
    nu = params[IDX['nu']]
    rho = params[IDX['rho']]
    v0 = params[IDX['v0']]
    lam = params[IDX['lam']]
    mu_J = params[IDX['mu_J']]
    sigma_J = params[IDX['sigma_J']]

    # domain checks
    if kappa <= 0 or theta <= 0 or nu <= 0 or v0 <= 0 or lam < 0:
        return 1e12

    # Feller condition (annual units) soft-penalty
    feller_gap = 2.0*kappa*theta - nu**2
    penalty = 0.0
    if feller_gap <= 0:
        penalty += 1e6 * (1.0 + (-feller_gap))

    # compute quasi-log-likelihood (sum of conditional normal logs)
    ll = 0.0
    v_prev = v0
    try:
        for r in returns:
            ll_step = step_loglik(r, v_prev, params)
            ll += ll_step
            # update expected variance for next step
            v_prev = theta + (v_prev - theta)*math.exp(-kappa*DT)
    except Exception:
        return 1e12

    # negative log-likelihood (we maximize ll)
    return -ll + penalty

# -----------------------
# Fitter function for a single series (prices or levels)
# -----------------------
def fit_series_to_bates(levels, name, console, with_jumps=WITH_JUMPS, restarts=NUM_OPTIMIZER_RESTARTS):
    # compute daily log returns
    log_returns = np.log(levels / levels.shift(1)).dropna().values
    n = len(log_returns)
    if n < 60:
        console.print(f"[bold yellow]Warning:[/bold yellow] Too few observations for '{name}' ({n} points). Skipping.")
        return None

    # annualize sample moments for sensible initial guesses
    mean_daily = np.mean(log_returns)
    var_daily = np.var(log_returns, ddof=0)
    mu_annual = mean_daily * days_in_year
    var_annual = var_daily * days_in_year
    vol_annual = math.sqrt(max(var_annual, 1e-12))

    rng = default_rng(123 + abs(hash(name)) % 9999)

    best_res = None
    best_val = np.inf

    # base initial param vector in ANNUAL units (length 9)
    init_base = np.zeros(9)
    init_base[IDX['mu']] = np.clip(mu_annual, BOUNDS[IDX['mu']][0], BOUNDS[IDX['mu']][1])
    init_base[IDX['kappa']] = 1.0
    init_base[IDX['theta']] = np.clip(var_annual, BOUNDS[IDX['theta']][0], BOUNDS[IDX['theta']][1])
    init_base[IDX['nu']] = 0.5
    init_base[IDX['rho']] = -0.3
    init_base[IDX['v0']] = np.clip(var_annual, BOUNDS[IDX['v0']][0], BOUNDS[IDX['v0']][1])
    init_base[IDX['lam']] = 0.5 if with_jumps else 0.0
    init_base[IDX['mu_J']] = -0.01
    init_base[IDX['sigma_J']] = 0.05

    # reduce bounds vector if we are Heston-only (still keep size 9, but lambda/muJ/sigJ locked if no jumps)
    bounds = list(BOUNDS)
    if not with_jumps:
        # set lambda/muJ/sigJ to fixed small ranges or zero
        bounds[IDX['lam']] = (0.0, 0.0)
        bounds[IDX['mu_J']] = (0.0, 0.0)
        bounds[IDX['sigma_J']] = (0.0, 0.0)

    # restart loop
    for attempt in range(restarts):
        console.print(f"[dim]Fitting {name}: restart {attempt+1}/{restarts} (with_jumps={with_jumps})[/dim]")
        # small randomized perturbation to init
        perturb = rng.normal(scale=[0.01, 0.5, max(1e-4, 0.5*init_base[IDX['theta']]), 0.2, 0.2, 0.5*init_base[IDX['v0']], 0.2, 0.005, 0.02])
        x0 = init_base + perturb

        # ensure within bounds
        for j in range(len(x0)):
            low, high = bounds[j]
            x0[j] = np.clip(x0[j], low + 1e-12, high - 1e-12)

        res = minimize(
            neg_log_likelihood,
            x0,
            args=(log_returns, with_jumps),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': MAXITER, 'ftol': 1e-8, 'disp': False}
        )

        if res.success and res.fun < best_val:
            best_val = res.fun
            best_res = res

    if best_res is None:
        console.print(f"[bold red]Failed to converge for {name}[/bold red]")
        return None

    params = best_res.x
    # package outputs (annual)
    mu = params[IDX['mu']]
    kappa = params[IDX['kappa']]
    theta = params[IDX['theta']]
    nu = params[IDX['nu']]
    rho = params[IDX['rho']]
    v0 = params[IDX['v0']]
    lam = params[IDX['lam']]
    muJ = params[IDX['mu_J']]
    sigmaJ = params[IDX['sigma_J']]

    out = {
        'name': name,
        'N_obs': n,
        'Drift (Annual mu)': mu,
        'kappa (1/yr)': kappa,
        'theta (annual var)': theta,
        'long_run_vol_%': math.sqrt(max(theta,1e-12))*100.0,
        'nu (vol-of-vol)': nu,
        'rho': rho,
        'v0 (annual var)': v0,
        'init_vol_%': math.sqrt(max(v0,1e-12))*100.0,
        'lambda (jumps/yr)': lam,
        'jump_mean (muJ)': muJ,
        'jump_std (sigmaJ)': sigmaJ,
        'NegLogLik': best_val
    }
    return out

# -----------------------
# Pretty table printer
# -----------------------
def print_rich_table(rows, title="Results"):
    if not rows:
        console.print("[bold red]No rows to print.[/bold red]")
        return
    df = pd.DataFrame(rows).set_index('name')
    table = Table(title=title, title_style="bold magenta", header_style="bold cyan")
    table.add_column("Name", justify="left", style="bold")
    for col in df.columns:
        table.add_column(col, justify="right")
    for idx, row in df.iterrows():
        formatted = []
        for v in row:
            if isinstance(v, (float, np.floating)):
                formatted.append(f"{v:.6f}")
            else:
                formatted.append(str(v))
        table.add_row(idx, *formatted)
    console.print(table)

# -----------------------
# Main runner
# -----------------------
def main():
    # load data
    if not os.path.exists(csv_filename):
        console.print(f"[bold red]Error:[/bold red] CSV '{csv_filename}' not found. Put the file in the working directory.")
        return

    data = pd.read_csv(csv_filename, index_col=0, parse_dates=True)
    # select only ASSET_NAMES that exist in the file
    cols = [c for c in ASSET_NAMES if c in data.columns]
    if not cols:
        console.print("[bold red]None of ASSET_NAMES found in CSV.[/bold red]")
        console.print(f"CSV columns: {list(data.columns[:50])}")
        return

    # normalize weights to present assets
    W_acc = np.zeros(len(cols))
    W_wdr = np.zeros(len(cols))
    for j, c in enumerate(cols):
        orig_idx = ASSET_NAMES.index(c)
        W_acc[j] = ACCUMULATION_WEIGHTS[orig_idx]
        W_wdr[j] = WITHDRAWAL_WEIGHTS[orig_idx]
    if W_acc.sum() > 0: W_acc /= W_acc.sum()
    if W_wdr.sum() > 0: W_wdr /= W_wdr.sum()

    levels = data[cols].copy()
    # ensure numeric and drop NA rows
    levels = levels.apply(pd.to_numeric, errors='coerce').dropna(how='all')
    # compute portfolio levels using log-linear aggregation (geometric weights)
    log_levels = np.log(levels)
    acc_level = np.exp((log_levels * W_acc).sum(axis=1))
    wdr_level = np.exp((log_levels * W_wdr).sum(axis=1))

    results = []

    # Fit Accumulation
    console.print("[bold green]Fitting Accumulation portfolio...[/bold green]")
    r = fit_series_to_bates(acc_level, "Accumulation", console, with_jumps=WITH_JUMPS)
    if r:
        print_rich_table([r], "Bates/Heston MLE - Accumulation")
        pd.DataFrame([r]).to_csv("Bates_MLE_Accumulation.csv", index=False)
        results.append(r)

    # Fit Withdrawal
    console.print("[bold green]Fitting Withdrawal portfolio...[/bold green]")
    r = fit_series_to_bates(wdr_level, "Withdrawal", console, with_jumps=WITH_JUMPS)
    if r:
        print_rich_table([r], "Bates/Heston MLE - Withdrawal")
        pd.DataFrame([r]).to_csv("Bates_MLE_Withdrawal.csv", index=False)
        results.append(r)

    # Fit individual assets
    for c in cols:
        console.print(f"[bold green]Fitting {c}...[/bold green]")
        r = fit_series_to_bates(levels[c], c, console, with_jumps=WITH_JUMPS)
        if r:
            print_rich_table([r], f"Bates/Heston MLE - {c}")
            safe_name = "".join(ch if ch.isalnum() or ch in "-_.() " else "_" for ch in c)
            pd.DataFrame([r]).to_csv(f"Bates_MLE_{safe_name}.csv", index=False)
            results.append(r)

    # Save combined results
    if results:
        pd.DataFrame(results).to_csv("Bates_MLE_Results.csv", index=False)
        console.print("[bold green]All done. Combined results saved to Bates_MLE_Results.csv[/bold green]")
    else:
        console.print("[bold yellow]No successful fits to save.[/bold yellow]")

if __name__ == "__main__":
    main()
