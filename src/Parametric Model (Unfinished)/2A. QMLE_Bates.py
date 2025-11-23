# ==============================================================
# VERSION 4.4 — Bates Per-Regime QMLE with Diagnostics & Plots
# ==============================================================
# This script fits a Bates (Heston + Poisson jumps) model to
# regime-classified returns using Quasi-Maximum Likelihood Estimation (QMLE).
# It generates parameter estimates, diagnostic statistics, and
# visualizations comparing empirical returns to the fitted Bates PDF.
#
# Key Features:
# - User-selectable regimes (REGIMES_TO_FIT)
# - True Bates QMLE with Gaussian step approximation
# - Optional bounds on parameters (USE_BOUNDS)
# - Density floor for numerical stability
# - Randomized restarts for robust convergence
# - Two Rich tables: 
#       (A) Model Parameters
#       (B) Diagnostics & Moments (log-likelihood, fitted mean/std)
# - Plots: Empirical histogram vs approximate Bates PDF for each fitted regime
# - Saves all results to CSV for further analysis
#
# Configuration Options:
# - CSV_FILE: Input CSV with regime classifications and returns
# - REGIME_COLUMN: Column name indicating regime ID
# - RETURNS_COLUMN: Column name of returns to fit
# - REGIMES_TO_FIT: List of regime IDs to fit
# - RETURNS_ARE_PERCENT: If True, divides CSV returns by 100
# - PERIODS_PER_YEAR / DT: Controls annualization of parameters
# - NUM_RESTARTS: Number of randomized restarts for optimizer
# - MAXITER: Maximum iterations per optimization attempt
# - USE_BOUNDS: Enable/disable parameter bounds during optimization
# - DENSITY_FLOOR: Minimum PDF density to avoid numerical issues
# - DEFAULT_BOUNDS: Parameter bounds for (mu, kappa, theta, nu, rho, v0, lam, mu_J, sigma_J)
# - PLOT_K_MAX: Maximum number of jumps in PDF approximation
#
# Dependencies:
#   pip install numpy pandas scipy matplotlib rich
#
# Notes:
# - Ensure the input CSV has the specified REGIME_COLUMN and RETURNS_COLUMN.
# - Percent returns in CSV are converted to decimals if RETURNS_ARE_PERCENT=True.
# - Fitted parameters are returned in annualized units.
# - Visualizations show empirical vs approximate Bates PDF.
# ----------------------------------------------------------------------------------------
# Author: VCODIO
# ----------------------------------------------------------------------------------------

import os
import math
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm, ks_2samp
from rich.console import Console
from rich.table import Table


# Configering 
CSV_FILE = "regime_classification_nominal_returns.csv"
REGIME_COLUMN = "Inferred Regime ID (0=Low, 2=High)"
RETURNS_COLUMN = "Total Nominal Return (%)"  # Updated to match actual CSV column name

# Which regimes to fit
REGIMES_TO_FIT = [0.0, 1.0, 2.0]

# Data frequency / units
RETURNS_ARE_PERCENT = True      # If True, will divide CSV returns by 100
PERIODS_PER_YEAR = 12            # 4=quarterly, 252=daily, 12=monthly
DT = 1.0 / PERIODS_PER_YEAR

# Fitting controls
NUM_RESTARTS = 15
MAXITER = 50000
USE_BOUNDS = True               # Set False to disable bounds
DENSITY_FLOOR = 1e-12
VERBOSE = True

# Bounds for (mu, kappa, theta, nu, rho, v0, lam, mu_J, sigma_J)
DEFAULT_BOUNDS = [
  (-0.5, 0.5),      # mu
    (0.1, 50.0),      # kappa
    (0.005, 1),     # theta
    (0.5, 10.0),      # nu <--- EXPANDED
    (-1.0, 0.0),      # rho
    (0.005, 1.0),     # v0
    (0.05, 20.0),     # lambda <--- EXPANDED
    (-5.0, -0.20),    # mu_J
    (0.2, 1.0),      # sigma_J
]

# Poisson truncation for the plots
PLOT_K_MAX = 40
console = Console(width=200)

# IDX mapping
IDX = {
    "mu": 0, "kappa": 1, "theta": 2, "nu": 3, "rho": 4, "v0": 5,
    "lam": 6, "mu_J": 7, "sigma_J": 8
}


# Math helpers
def norm_logpdf(x, mean, sd):
    if sd <= 0 or not math.isfinite(sd):
        return -1e8
    return -0.5 * math.log(2 * math.pi) - math.log(sd) - 0.5 * ((x - mean) / sd) ** 2


# Step log-likelihood (QMLE) using Heston expectation + Gaussian jumps approx
def step_loglik(r_t, v_prev, params):
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

    # Jump mean & variance contribution in dt
    mJ = lam * dt * mu_J
    sJ2 = lam * dt * (sigma_J ** 2 + mu_J ** 2)

    # Exact expectation of v_t under CIR:
    Ev_t = theta + (v_prev - theta) * math.exp(-kappa * dt)
    phi = (1 - math.exp(-kappa * dt))
    # trapezoid proxy for average variance across interval
    v_bar = (1 - 0.5 * phi) * v_prev + 0.5 * phi * Ev_t

    mean_r = mu * dt + mJ
    var_r = v_bar * dt + sJ2

    if not math.isfinite(var_r) or var_r <= 0:
        return -1e8
    sd = math.sqrt(var_r)
    return norm_logpdf(r_t, mean_r, sd)


# Full negative log-likelihood across returns (params in annual units)
def neg_log_likelihood(params, returns, with_jumps=True):
    # domain checks
    if np.any(~np.isfinite(params)):
        return 1e12
    kappa = params[IDX['kappa']]
    theta = params[IDX['theta']]
    nu = params[IDX['nu']]
    v0 = params[IDX['v0']]
    lam = params[IDX['lam']]
    if kappa <= 0 or theta <= 0 or nu <= 0 or v0 <= 0 or lam < 0:
        return 1e12

    # Feller condition soft penalty
    feller_gap = 2.0 * kappa * theta - nu ** 2
    penalty = 0.0
    if feller_gap <= 0:
        penalty += 1e6 * (1.0 + (-feller_gap))

    ll = 0.0
    v_prev = params[IDX['v0']]
    try:
        for r in returns:
            ll_step = step_loglik(r, v_prev, params)
            ll += ll_step
            v_prev = params[IDX['theta']] + (v_prev - params[IDX['theta']]) * math.exp(-params[IDX['kappa']] * DT)
    except Exception:
        return 1e12

    return -ll + penalty


# Calculating the per step loglik stats
def compute_loglik_stats(params, returns):
    v_prev = params[IDX['v0']]
    pdfs = []
    for r in returns:
        ll = step_loglik(r, v_prev, params)
        # turn logpdf to pdf with floor
        p = math.exp(ll) if ll > -1e8 else DENSITY_FLOOR
        pdfs.append(max(p, DENSITY_FLOOR))
        # propagate
        v_prev = params[IDX['theta']] + (v_prev - params[IDX['theta']]) * math.exp(-params[IDX['kappa']] * DT)
    pdfs = np.array(pdfs)
    ll_arr = np.log(pdfs)
    return float(ll_arr.mean()), float(ll_arr.sum())


# Approximate Bates PDF for the visualization

def approx_bates_pdf(x, mu, theta, lam, mu_J, sigma_J, k_max=PLOT_K_MAX):
    xarr = np.atleast_1d(x)
    pdf = np.zeros_like(xarr, dtype=float)
    var_diffusion = theta * DT
    for k in range(k_max + 1):
        weight = math.exp(-lam * DT) * ((lam * DT) ** k) / math.factorial(k)
        mean_k = mu * DT + k * mu_J
        var_k = var_diffusion + k * (sigma_J ** 2)
        std_k = math.sqrt(max(var_k, 1e-16))
        pdf += weight * norm.pdf(xarr, loc=mean_k, scale=std_k)
    return pdf


# Fitting wrapper per regime
def fit_bates_to_returns(returns, name="regime", with_jumps=True,
                         restarts=NUM_RESTARTS, maxiter=MAXITER,
                         bounds_vec=DEFAULT_BOUNDS, use_bounds=USE_BOUNDS):
    n = len(returns)
    if n < 8:
        console.print(f"[yellow]Too few observations for {name} ({n}); skipping.[/yellow]")
        return None

    # empirical per-period moments -> annualize for initial guesses
    mean_period = np.mean(returns)
    var_period = np.var(returns, ddof=0)
    mu_annual = mean_period / DT
    var_annual = var_period / DT
    vol_annual = math.sqrt(max(var_annual, 1e-12))

    # base initial vector
    init_base = np.zeros(9, dtype=float)
    init_base[IDX['mu']] = np.clip(mu_annual, bounds_vec[IDX['mu']][0], bounds_vec[IDX['mu']][1])
    init_base[IDX['kappa']] = 1.0
    init_base[IDX['theta']] = np.clip(var_annual, bounds_vec[IDX['theta']][0], bounds_vec[IDX['theta']][1])
    init_base[IDX['nu']] = 0.5
    init_base[IDX['rho']] = -0.3
    init_base[IDX['v0']] = np.clip(var_annual, bounds_vec[IDX['v0']][0], bounds_vec[IDX['v0']][1])
    init_base[IDX['lam']] = 0.5 if with_jumps else 0.0
    init_base[IDX['mu_J']] = -0.01
    init_base[IDX['sigma_J']] = 0.05

    # bounds handling
    if use_bounds:
        bounds = list(bounds_vec)
    else:
        bounds = [(None, None)] * len(init_base)

    best_val = np.inf
    best_res = None
    rng = np.random.default_rng(12345 + abs(hash(name)) % 9999)

    for attempt in range(restarts):
        if VERBOSE:
            console.print(f"[dim]Fitting {name}: restart {attempt+1}/{restarts}[/dim]")
        scales = np.array([0.01, 0.5, max(1e-6, 0.5 * init_base[IDX['theta']]), 0.2, 0.1,
                           0.5 * max(1e-6, init_base[IDX['v0']]), 0.2, 0.005, 0.02])
        perturb = rng.normal(scale=scales)
        x0 = init_base + perturb

        if use_bounds:
            for j in range(len(x0)):
                lo, hi = bounds[j]
                if lo is not None: x0[j] = max(x0[j], lo + 1e-12)
                if hi is not None: x0[j] = min(x0[j], hi - 1e-12)

        res = minimize(neg_log_likelihood, x0,
                       args=(returns, with_jumps),
                       method="L-BFGS-B",
                       bounds=bounds if use_bounds else None,
                       options={"maxiter": maxiter, "ftol": 1e-8, "disp": False})

        if res.success and res.fun < best_val:
            best_val = res.fun
            best_res = res

    if best_res is None:
        console.print(f"[red]Failed to converge for {name}[/red]")
        return None

    params = best_res.x
    mu = params[IDX['mu']]; kappa = params[IDX['kappa']]; theta = params[IDX['theta']]
    nu = params[IDX['nu']]; rho = params[IDX['rho']]; v0 = params[IDX['v0']]
    lam = params[IDX['lam']]; muJ = params[IDX['mu_J']]; sigmaJ = params[IDX['sigma_J']]

    mean_ll, total_ll = compute_loglik_stats(params, returns)

    # approximate model mean & std per period using long-run theta
    mean_fit_period = mu * DT + lam * DT * muJ
    var_fit_period = theta * DT + lam * DT * (sigmaJ ** 2 + muJ ** 2)
    std_fit_period = math.sqrt(max(var_fit_period, 0.0))

    # check bounds hit
    bounds_hit = False
    hit_info = []
    if use_bounds:
        for i, val in enumerate(params):
            lo, hi = bounds_vec[i]
            if lo is not None and abs(val - lo) < 1e-8:
                bounds_hit = True
                hit_info.append((i, 'low', val))
            if hi is not None and abs(val - hi) < 1e-8:
                bounds_hit = True
                hit_info.append((i, 'high', val))

    out = {
        "name": name,
        "N_obs": n,
        "mu_annual": mu,
        "kappa": kappa,
        "theta_annual": theta,
        "nu": nu,
        "rho": rho,
        "v0": v0,
        "lambda_per_year": lam,
        "mu_J": muJ,
        "sigma_J": sigmaJ,
        "mean_ll_per_obs": mean_ll,
        "total_negloglik": best_val,
        "model_mean_per_period": mean_fit_period,
        "model_std_per_period": std_fit_period,
        "bounds_hit": bounds_hit,
        "bounds_hit_info": hit_info
    }
    return out

# -------------------------
# Plotting function
# -------------------------
def plot_empirical_vs_bates(returns, params, regime_id, save_fig=True, output_dir="output"):
    mu = params[IDX['mu']]
    theta = params[IDX['theta']]
    lam = params[IDX['lam']]
    muJ = params[IDX['mu_J']]
    sigmaJ = params[IDX['sigma_J']]

    xs = np.linspace(min(returns), max(returns), 400)
    ys = approx_bates_pdf(xs, mu, theta, lam, muJ, sigmaJ, k_max=PLOT_K_MAX)

    plt.figure(figsize=(8,5))
    plt.hist(returns, bins=30, density=True, alpha=0.6, color='gray', label='Empirical')
    plt.plot(xs, ys, 'r-', lw=2, label='Bates (approx) PDF')
    plt.title(f"Regime {regime_id}: Empirical vs Bates (approx) PDF")
    plt.xlabel("Return (decimal per period)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    if save_fig:
        os.makedirs(output_dir, exist_ok=True)
        safe = str(regime_id).replace(" ", "_")
        plt.savefig(os.path.join(output_dir, f"bates_regime_{safe}.png"), dpi=150, bbox_inches="tight")
    plt.close()


# Two-table printing helpers
def print_parameter_table(rows):
    if not rows:
        console.print("[yellow]No parameter rows to show[/yellow]")
        return
    table = Table(title="Bates Fit Parameters", header_style="bold cyan", show_lines=True)
    table.add_column("name", style="bold")
    cols = ["N_obs", "mu_annual", "kappa", "theta_annual", "nu", "rho", "v0", "lambda_per_year", "mu_J", "sigma_J"]
    for c in cols:
        table.add_column(c, justify="right", overflow="fold")
    for r in rows:
        table.add_row(
            r["name"],
            str(r["N_obs"]),
            f"{r['mu_annual']:.6f}",
            f"{r['kappa']:.6f}",
            f"{r['theta_annual']:.6f}",
            f"{r['nu']:.6f}",
            f"{r['rho']:.6f}",
            f"{r['v0']:.6f}",
            f"{r['lambda_per_year']:.6f}",
            f"{r['mu_J']:.6f}",
            f"{r['sigma_J']:.6f}"
        )
    console.print(table)

def print_diagnostics_table(rows):
    if not rows:
        console.print("[yellow]No diagnostics rows to show[/yellow]")
        return
    table = Table(title="Fit Diagnostics & Moments", header_style="bold magenta", show_lines=True)
    table.add_column("name", style="bold")
    diag_cols = ["mean_ll_per_obs", "total_negloglik", "model_mean_per_period", "model_std_per_period", "bounds_hit"]
    for c in diag_cols:
        table.add_column(c, justify="right", overflow="fold")
    for r in rows:
        table.add_row(
            r["name"],
            f"{r['mean_ll_per_obs']:.6f}",
            f"{r['total_negloglik']:.6f}",
            f"{r['model_mean_per_period']:.6e}",
            f"{r['model_std_per_period']:.6e}",
            str(r["bounds_hit"])
        )
    console.print(table)

# MAIN
def main():
    if not os.path.exists(CSV_FILE):
        console.print(f"[red]CSV file not found: {CSV_FILE}[/red]")
        return

    df = pd.read_csv(CSV_FILE)
    console.print(f"[debug] CSV columns: {list(df.columns)}")
    console.print(f"[debug] CSV head:\n{df.head()}")

    if REGIME_COLUMN not in df.columns or RETURNS_COLUMN not in df.columns:
        console.print("[red]CSV missing required columns. Check REGIME_COLUMN and RETURNS_COLUMN settings.[/red]")
        return

    results = []
    plotting_data = []  # Store data for plotting at the end
    
    for r in REGIMES_TO_FIT:
        # select rows matching regime; allow numeric or string matching
        mask = df[REGIME_COLUMN] == r
        if mask.sum() == 0:
            # try loose match (e.g., ints vs floats)
            mask = df[REGIME_COLUMN].astype(str) == str(r)
        regime_vals = df.loc[mask, RETURNS_COLUMN].values.astype(float)
        if len(regime_vals) == 0:
            console.print(f"[yellow]No data for regime {r}; skipping.[/yellow]")
            continue

        # convert percent->decimal if flagged
        returns = regime_vals / 100.0 if RETURNS_ARE_PERCENT else regime_vals.copy()

        console.print(f"[green]Fitting regime {r} with {len(returns)} observations[/green]")
        fit = fit_bates_to_returns(returns, name=f"regime_{r}", with_jumps=True,
                                   restarts=NUM_RESTARTS, maxiter=MAXITER,
                                   bounds_vec=DEFAULT_BOUNDS, use_bounds=USE_BOUNDS)
        if fit:
            results.append(fit)
            # Prepare params vector for plotting approx
            params_vec = np.zeros(9)
            params_vec[IDX['mu']] = fit['mu_annual']
            params_vec[IDX['theta']] = fit['theta_annual']
            params_vec[IDX['lam']] = fit['lambda_per_year']
            params_vec[IDX['mu_J']] = fit['mu_J']
            params_vec[IDX['sigma_J']] = fit['sigma_J']
            # Store for plotting at the end (don't plot now)
            plotting_data.append({
                'returns': returns,
                'params': params_vec,
                'regime_id': r
            })

    # Print results tables first
    if results:
        print_parameter_table(results)
        print_diagnostics_table(results)
        pd.DataFrame(results).to_csv("Bates_per_regime_QMLE_results.csv", index=False)
        console.print("[green]Saved Bates_per_regime_QMLE_results.csv[/green]")
        
        # Now plot all images at the end (non-blocking)
        console.print("\n[cyan]Generating plots for all regimes...[/cyan]")
        for plot_data in plotting_data:
            plot_empirical_vs_bates(
                plot_data['returns'],
                plot_data['params'],
                plot_data['regime_id'],
                save_fig=True
            )
        console.print("[green]✓ All plots saved as PNG files in output/ directory[/green]")
    else:
        console.print("[yellow]No successful fits.[/yellow]")

if __name__ == "__main__":
    main()
