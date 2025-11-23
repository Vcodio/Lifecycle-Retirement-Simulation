# ============================================================
# Bates Monte Carlo Simulator
#
# Simple script that compares the historical CAGR/Vol
# to simulated portfolios under Bates model.
# ============================================================

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
import matplotlib.pyplot as plt
import matplotlib.style as style

console = Console()
days_in_year = 252
csv_filename = 'Monte Carlo Asset Analyzer.csv'

# ============================================================
# PARAMETERS
# ============================================================

params = {
    "mu": 0.1437,         # Annual drift
    "kappa": 0.4159,      # Mean reversion speed (1/year)
    "theta": 0.0276,      # Long-run variance (annualized)
    "nu": 0.028,          # Vol of vol (annualized)
    "rho": -0.0949,       # Correlation
    "v0": 0.0039,         # Initial variance (annualized)
    "lam": 0.2635,        # Jump intensity (per year)
    "mu_J": -0.1364,      # Jump mean (decimal, e.g. -0.136 = -13.6%)
    "sigma_J": 0.0536,    # Jump std dev (decimal)
}

NUM_SIMULATIONS = 10
np.random.seed()

ASSET_NAMES = ['VTISIM','SPYSIM?L=2','VXUSSIM','DFSVX','DISVX','ZROZSIM']
ACCUMULATION_WEIGHTS = np.array([0.3, 0.2, 0.15, 0.10, 0.05, 0.2])

# ============================================================
# Simulation Function
# ============================================================
def simulate_bates(S0, T, N, params):
    dt = T / N
    mu, kappa, theta, nu, rho, v0, lam, mu_J, sigma_J = (
        params['mu'], params['kappa'], params['theta'], params['nu'], params['rho'],
        params['v0'], params['lam'], params['mu_J'], params['sigma_J']
    )

    price_path = np.zeros(N + 1)
    var_path = np.zeros(N + 1)
    price_path[0] = S0
    var_path[0] = v0

    for i in range(N):
        z1 = np.random.standard_normal()
        z2 = rho * z1 + np.sqrt(1 - rho**2) * np.random.standard_normal()

        # Heston variance
        var_path[i+1] = var_path[i] + kappa*(theta - var_path[i])*dt + nu*np.sqrt(np.maximum(var_path[i],0))*np.sqrt(dt)*z2
        var_path[i+1] = np.maximum(var_path[i+1], 1e-12)

        # Merton jumps
        num_jumps = np.random.poisson(lam * dt)
        jump_size = np.sum(np.random.normal(mu_J, sigma_J, num_jumps)) if num_jumps > 0 else 0.0

        price_path[i+1] = price_path[i] * np.exp((mu - 0.5*var_path[i])*dt + np.sqrt(np.maximum(var_path[i],0))*np.sqrt(dt)*z1 + jump_size)

    return pd.DataFrame({"Price": price_path, "Variance": var_path})

# ============================================================
# Helper Functions
# ============================================================
def annualized_cagr(price_series):
    n_years = len(price_series)/days_in_year
    return (price_series[-1]/price_series[0])**(1/n_years) - 1

def annualized_vol(price_series):
    log_returns = np.log(price_series[1:]/price_series[:-1])
    return np.std(log_returns)*np.sqrt(days_in_year)

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    console.print(f"[bold green]Starting Bates Monte Carlo Simulation with {NUM_SIMULATIONS} paths...[/bold green]")

    # Load historical portfolio
    try:
        data = pd.read_csv(csv_filename, index_col=0, parse_dates=True)
        cols = [c for c in ASSET_NAMES if c in data.columns]
        W_acc = np.zeros(len(cols))
        for j, c in enumerate(cols):
            orig_idx = ASSET_NAMES.index(c)
            W_acc[j] = ACCUMULATION_WEIGHTS[orig_idx]
        if W_acc.sum() > 0: W_acc /= W_acc.sum()

        log_levels = np.log(data[cols])
        acc_level = np.exp((log_levels * W_acc).sum(axis=1))

        T = (len(acc_level)-1)/days_in_year
        N = len(acc_level)-1
        S0 = acc_level.iloc[0]

    except FileNotFoundError:
        console.print(f"[bold red]Error:[/bold red] '{csv_filename}' not found.")
        acc_level = None
        T, N, S0 = 1.0, 252, 100.0

    # Run simulations
    style.use('dark_background')
    plt.figure(figsize=(12,8))

    simulated_paths = []
    for i in range(NUM_SIMULATIONS):
        sim_df = simulate_bates(S0, T, N, params)
        if acc_level is not None:
            sim_df.index = acc_level.index
        plt.plot(sim_df.index, sim_df['Price'], alpha=0.5, color='green')
        simulated_paths.append(sim_df['Price'].values)

    # Plot historical portfolio
    if acc_level is not None:
        plt.plot(acc_level.index, acc_level.values, color='red', linewidth=3, label='Historical Portfolio')

    # Compute metrics
    simulated_paths = np.array(simulated_paths)
    sim_cagrs = np.array([annualized_cagr(p) for p in simulated_paths])
    sim_vols = np.array([annualized_vol(p) for p in simulated_paths])

    # Historical metrics
    if acc_level is not None:
        hist_cagr = annualized_cagr(acc_level.values)
        hist_vol = annualized_vol(acc_level.values)
    else:
        hist_cagr, hist_vol = np.nan, np.nan

    # Percentiles
    percentiles = [10,25,50,75,90]
    sim_cagr_pct = np.percentile(sim_cagrs, percentiles)
    sim_vol_pct = np.percentile(sim_vols, percentiles)

    # Display table
    stats_table = Table(title="CAGR / Vol Comparison", header_style="bold cyan")
    stats_table.add_column("Metric", justify="left", style="bold")
    stats_table.add_column("Historical", justify="right")
    for p in percentiles:
        stats_table.add_column(f"{p}%", justify="right")

    stats_table.add_row("CAGR", f"{hist_cagr:.2%}", *[f"{v:.2%}" for v in sim_cagr_pct])
    stats_table.add_row("Vol", f"{hist_vol:.2%}", *[f"{v:.2%}" for v in sim_vol_pct])
    console.print(stats_table)

    plt.title('Bates Model Monte Carlo Simulation (Annual Params)', color='white')
    plt.xlabel('Date', color='white')
    plt.ylabel('Asset Price', color='white')
    plt.grid(True)
    plt.legend()
    plt.show()
