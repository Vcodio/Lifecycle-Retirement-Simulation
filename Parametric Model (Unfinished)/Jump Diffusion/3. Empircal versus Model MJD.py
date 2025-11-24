"""
Monte Carlo Simulation: Regime-Switching Merton Jump Diffusion (MJD) Model

This script runs Monte Carlo simulations using:
1. Merton Jump Diffusion model parameters fitted to each regime (from moment matching)
2. HMM transition probabilities for regime switching

It compares empirical data statistics to simulated model statistics:
- Median CAGR (Compound Annual Growth Rate)
- Median Annual Volatility
- Median Skewness
- Median Kurtosis
- Median Max Drawdown

MJD Model:
- Geometric Brownian Motion (constant volatility) + Merton jumps
- Simpler than Bates model (no stochastic volatility)
- Parameters: mu (drift), sigma (volatility), lambda (jump intensity), mu_J (jump mean), sigma_J (jump volatility)

Usage:
    python "3. Empircal versus Model MJD.py" [--n-simulations N] [--n-periods N] [--seed N] [--csv PATH]
"""

import os
import sys
import math
import numpy as np
import pandas as pd
from scipy import stats
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Constants
PERIODS_PER_YEAR = 12
DT = 1.0 / PERIODS_PER_YEAR

# Parameter indices for MJD: (mu, sigma, lam, mu_J, sigma_J)
MU_IDX = 0
SIGMA_IDX = 1
LAM_IDX = 2
MU_J_IDX = 3
SIGMA_J_IDX = 4

IDX = {
    "mu": MU_IDX,
    "sigma": SIGMA_IDX,
    "lam": LAM_IDX,
    "mu_J": MU_J_IDX,
    "sigma_J": SIGMA_J_IDX
}

console = Console(width=120)

# ============================================================================
# DATA LOADING
# ============================================================================

def load_mjd_parameters(csv_path):
    """Load Merton Jump Diffusion model parameters for each regime from CSV.
    
    Returns:
        dict: {regime_id: params_array} where params_array is 5-element array
        dict: {regime_id: moment_matching_info} with fitting quality metrics
    """
    df = pd.read_csv(csv_path)
    params_dict = {}
    moment_matching_info = {}
    
    for _, row in df.iterrows():
        regime_id = float(row['name'].split('_')[1])
        params = np.array([
            row['mu_annual'],
            row['sigma_annual'],
            row['lambda_per_year'],
            row['mu_J'],
            row['sigma_J']
        ])
        params_dict[regime_id] = params
        
        # Store moment matching quality metrics
        moment_matching_info[regime_id] = {
            'n_obs': int(row['N_obs']),
            'objective': row['objective'],
            'emp_mean': row['emp_mean'],
            'emp_std': row['emp_std'],
            'emp_skew': row['emp_skew'],
            'emp_kurt': row['emp_kurt'],
            'emp_max_dd': row['emp_max_dd'],
            'model_mean': row['model_mean'],
            'model_std': row['model_std'],
            'model_skew': row['model_skew'],
            'model_kurt': row['model_kurt'],
            'model_max_dd': row['model_max_dd'],
            'mean_error_pct': row['mean_error_pct'],
            'std_error_pct': row['std_error_pct'],
            'skew_error_pct': row['skew_error_pct'],
            'kurt_error_pct': row['kurt_error_pct'],
            'max_dd_error_pct': row['max_dd_error_pct']
        }
    
    return params_dict, moment_matching_info

def load_transition_matrix(csv_path):
    """Load HMM transition matrix from CSV.
    
    Returns:
        np.array: Transition matrix (n_regimes x n_regimes)
        list: Regime IDs in order
    """
    df = pd.read_csv(csv_path, index_col=0)
    
    # Extract regime IDs from column names
    regime_ids = []
    for col in df.columns:
        # Extract number from "0: V-1: Lowest Volatility" format
        regime_id = float(col.split(':')[0].strip())
        regime_ids.append(regime_id)
    
    # Build transition matrix - ensure rows and columns are in same order
    n_regimes = len(regime_ids)
    transition_matrix = np.zeros((n_regimes, n_regimes))
    
    # Create mapping from regime ID to index
    regime_to_idx = {regime_id: idx for idx, regime_id in enumerate(regime_ids)}
    
    for i, row_name in enumerate(df.index):
        # Extract regime ID from row name
        from_regime = float(row_name.split(':')[0].strip())
        from_idx = regime_to_idx[from_regime]
        
        for j, col_name in enumerate(df.columns):
            to_regime = float(col_name.split(':')[0].strip())
            to_idx = regime_to_idx[to_regime]
            transition_matrix[from_idx, to_idx] = df.iloc[i, j]
    
    return transition_matrix, regime_ids

def load_empirical_data(csv_path):
    """Load empirical returns data.
    
    Returns:
        np.array: Log returns
        np.array: Regime IDs (for reference)
    """
    df = pd.read_csv(csv_path)
    
    # Get returns column
    if 'Total Nominal Return (%)' in df.columns:
        returns = df['Total Nominal Return (%)'].values / 100.0
    elif 'returns' in df.columns:
        returns = df['returns'].values
    else:
        raise ValueError("Could not find returns column")
    
    # Convert to log returns if needed (check if already log returns)
    if np.any(np.abs(returns) > 0.5):
        returns = np.log(1.0 + returns)
    
    # Get regime IDs if available
    regime_col = None
    for col in df.columns:
        if 'regime' in col.lower() or 'Regime' in col:
            regime_col = col
            break
    
    regime_ids = df[regime_col].values if regime_col else None
    
    return returns, regime_ids

# ============================================================================
# MERTON JUMP DIFFUSION MODEL SIMULATION
# ============================================================================

def simulate_mjd_period(params, rng):
    """Simulate one period of Merton Jump Diffusion model.
    
    Args:
        params: 5-element array of MJD parameters [mu, sigma, lam, mu_J, sigma_J]
        rng: Random number generator
    
    Returns:
        log_return: Log return for this period
    """
    mu = params[IDX['mu']]
    sigma = params[IDX['sigma']]
    lam = params[IDX['lam']]
    mu_J = params[IDX['mu_J']]
    sigma_J = params[IDX['sigma_J']]
    
    # Standard Wiener process
    z = rng.standard_normal()
    
    # Merton jumps
    num_jumps = rng.poisson(lam * DT)
    if num_jumps > 0:
        jump_size = np.sum(rng.normal(mu_J, sigma_J, num_jumps))
    else:
        jump_size = 0.0
    
    # MJD price process: dS/S = mu*dt + sigma*dW + dJ
    log_return = (mu - 0.5 * sigma ** 2) * DT + sigma * math.sqrt(DT) * z + jump_size
    
    return log_return

def simulate_regime_switching_mjd(params_dict, transition_matrix, regime_ids, n_periods, 
                                  initial_regime=None, seed=None, burn_in_periods=0):
    """Simulate regime-switching Merton Jump Diffusion model.
    
    Args:
        params_dict: {regime_id: params_array} for each regime
        transition_matrix: Transition probability matrix (n_regimes x n_regimes)
        regime_ids: List of regime IDs in order matching transition_matrix
        n_periods: Number of periods to simulate
        initial_regime: Initial regime ID (default: sample from stationary distribution)
        seed: Random seed
        burn_in_periods: Number of periods to simulate before starting to record (for equilibrium)
    
    Returns:
        returns: Array of log returns (n_periods,)
        regimes: Array of regime IDs (n_periods,)
    """
    rng = np.random.default_rng(seed)
    
    # Initialize regime
    if initial_regime is None:
        # Sample from stationary distribution
        # Compute stationary distribution as left eigenvector of transition matrix
        eigenvals, eigenvecs = np.linalg.eig(transition_matrix.T)
        stationary_idx = np.argmax(np.real(eigenvals))
        stationary_dist = np.real(eigenvecs[:, stationary_idx])
        stationary_dist = stationary_dist / np.sum(stationary_dist)
        initial_regime_idx = rng.choice(len(regime_ids), p=stationary_dist)
        current_regime = regime_ids[initial_regime_idx]
    else:
        current_regime = initial_regime
    
    # Initialize arrays
    returns = np.zeros(n_periods)
    regimes = np.zeros(n_periods)
    
    # Burn-in period (simulate but don't record)
    for _ in range(burn_in_periods):
        params = params_dict[current_regime]
        _ = simulate_mjd_period(params, rng)
        
        # Transition to next regime
        current_regime_idx = regime_ids.index(current_regime)
        transition_probs = transition_matrix[current_regime_idx, :]
        next_regime_idx = rng.choice(len(regime_ids), p=transition_probs)
        current_regime = regime_ids[next_regime_idx]
    
    # Main simulation
    for t in range(n_periods):
        # Store current regime
        regimes[t] = current_regime
        
        # Get parameters for current regime
        params = params_dict[current_regime]
        
        # Simulate one period
        log_return = simulate_mjd_period(params, rng)
        returns[t] = log_return
        
        # Transition to next regime
        current_regime_idx = regime_ids.index(current_regime)
        transition_probs = transition_matrix[current_regime_idx, :]
        next_regime_idx = rng.choice(len(regime_ids), p=transition_probs)
        current_regime = regime_ids[next_regime_idx]
    
    return returns, regimes

# ============================================================================
# SIMULATION-BASED MOMENT COMPUTATION
# ============================================================================

def compute_moments_via_simulation(params, n_simulations=10000, n_periods=1000, 
                                    burn_in_periods=100, seed=None):
    """Compute MJD model moments via Monte Carlo simulation.
    
    This is an alternative to analytical formulas that ensures exact matching
    between parameter fitting and simulation behavior.
    
    Args:
        params: 5-element array of MJD parameters
        n_simulations: Number of simulation paths
        n_periods: Number of periods per path
        burn_in_periods: Burn-in periods (not needed for MJD but kept for consistency)
        seed: Random seed
    
    Returns:
        dict with keys: 'mean', 'std', 'skew', 'kurt', 'max_dd'
    """
    rng = np.random.default_rng(seed)
    all_returns = []
    
    for sim in range(n_simulations):
        # Single regime simulation (no switching)
        returns = []
        
        # Burn-in (not needed for MJD but kept for consistency)
        for _ in range(burn_in_periods):
            _ = simulate_mjd_period(params, rng)
        
        # Main simulation
        for _ in range(n_periods):
            log_return = simulate_mjd_period(params, rng)
            returns.append(log_return)
        
        all_returns.extend(returns)
    
    all_returns = np.array(all_returns)
    
    # Compute statistics
    mean_period = np.mean(all_returns)
    std_period = np.std(all_returns, ddof=0)
    skew = stats.skew(all_returns, bias=False) if len(all_returns) > 2 else 0.0
    kurt = stats.kurtosis(all_returns, fisher=True, bias=False) if len(all_returns) > 3 else 0.0
    
    # Max drawdown (compute on a per-path basis, then average)
    max_dds = []
    for sim in range(n_simulations):
        path_returns = all_returns[sim * n_periods:(sim + 1) * n_periods]
        if len(path_returns) > 0:
            cumulative_price = np.cumprod(1.0 + path_returns)
            peak_series = np.maximum.accumulate(cumulative_price)
            drawdowns = (cumulative_price - peak_series) / peak_series
            max_dd = np.min(drawdowns) if len(drawdowns) > 0 else 0.0
            max_dds.append(max_dd)
    
    max_dd = np.mean(max_dds) if len(max_dds) > 0 else 0.0
    
    return {
        'mean': mean_period,
        'std': std_period,
        'skew': skew,
        'kurt': kurt,
        'max_dd': max_dd
    }

# ============================================================================
# STATISTICS COMPUTATION
# ============================================================================

def compute_cagr(returns):
    """Compute Compound Annual Growth Rate from log returns.
    
    Args:
        returns: Array of log returns
    
    Returns:
        CAGR: Annualized return
    """
    if len(returns) == 0:
        return 0.0
    total_return = np.sum(returns)
    n_years = len(returns) / PERIODS_PER_YEAR
    if n_years > 0:
        cagr = total_return / n_years
    else:
        cagr = 0.0
    return cagr

def compute_annual_volatility(returns):
    """Compute annualized volatility from log returns.
    
    Args:
        returns: Array of log returns
    
    Returns:
        Annualized volatility
    """
    if len(returns) < 2:
        return 0.0
    period_vol = np.std(returns, ddof=0)
    annual_vol = period_vol * math.sqrt(PERIODS_PER_YEAR)
    return annual_vol

def compute_skewness(returns):
    """Compute skewness from log returns.
    
    Args:
        returns: Array of log returns
    
    Returns:
        Skewness
    """
    if len(returns) < 3:
        return 0.0
    return stats.skew(returns, bias=False)

def compute_kurtosis(returns):
    """Compute excess kurtosis from log returns.
    
    Args:
        returns: Array of log returns
    
    Returns:
        Excess kurtosis
    """
    if len(returns) < 4:
        return 0.0
    return stats.kurtosis(returns, fisher=True, bias=False)

def compute_max_drawdown(returns):
    """Compute maximum drawdown from log returns.
    
    Args:
        returns: Array of log returns
    
    Returns:
        Maximum drawdown (negative value)
    """
    if len(returns) == 0:
        return 0.0
    
    # Convert to cumulative price
    cumulative_price = np.cumprod(1.0 + returns)
    peak_series = np.maximum.accumulate(cumulative_price)
    drawdowns = (cumulative_price - peak_series) / peak_series
    max_dd = np.min(drawdowns) if len(drawdowns) > 0 else 0.0
    return max_dd

def compute_statistics(returns):
    """Compute all statistics for a returns series.
    
    Returns:
        dict with keys: 'cagr', 'vol', 'skew', 'kurt', 'max_dd'
    """
    return {
        'cagr': compute_cagr(returns),
        'vol': compute_annual_volatility(returns),
        'skew': compute_skewness(returns),
        'kurt': compute_kurtosis(returns),
        'max_dd': compute_max_drawdown(returns)
    }

# ============================================================================
# MAIN SIMULATION
# ============================================================================

def run_monte_carlo_simulation(params_dict, transition_matrix, regime_ids, 
                               n_simulations=1000, n_periods=None, seed=None, 
                               return_paths=False):
    """Run Monte Carlo simulation with regime switching.
    
    Args:
        params_dict: {regime_id: params_array} for each regime
        transition_matrix: Transition probability matrix
        regime_ids: List of regime IDs
        n_simulations: Number of simulation paths
        n_periods: Number of periods per simulation (if None, use empirical length)
        seed: Random seed
        return_paths: If True, also return returns and regimes arrays
    
    Returns:
        list of dicts: Each dict contains statistics for one simulation
        (if return_paths=True, also returns list of (returns, regimes) tuples)
    """
    if n_periods is None:
        # Use average regime duration to estimate length
        # For now, use a reasonable default (e.g., 1000 periods = ~83 years)
        n_periods = 1000
    
    all_stats = []
    all_paths = [] if return_paths else None
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TextColumn("[cyan]{task.fields[current_sim]}/{task.fields[total_sims]}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task(
            "[bold cyan]Running Monte Carlo simulations...",
            total=n_simulations,
            current_sim=0,
            total_sims=n_simulations
        )
        
        for sim in range(n_simulations):
            # Use different seed for each simulation
            sim_seed = seed + sim if seed is not None else None
            returns, regimes = simulate_regime_switching_mjd(
                params_dict, transition_matrix, regime_ids, n_periods,
                seed=sim_seed
            )
            
            stats_dict = compute_statistics(returns)
            all_stats.append(stats_dict)
            
            if return_paths:
                all_paths.append((returns, regimes))
            
            progress.update(task, advance=1, current_sim=sim + 1)
    
    if return_paths:
        return all_stats, all_paths
    return all_stats

# ============================================================================
# COMPARISON AND OUTPUT
# ============================================================================

def compare_empirical_vs_model(empirical_returns, simulated_stats):
    """Compare empirical statistics to simulated model statistics.
    
    Args:
        empirical_returns: Array of empirical log returns
        simulated_stats: List of dicts with statistics from simulations
    
    Returns:
        dict with comparison results
    """
    # Compute empirical statistics
    emp_stats = compute_statistics(empirical_returns)
    
    # Extract simulated statistics
    sim_cagrs = [s['cagr'] for s in simulated_stats]
    sim_vols = [s['vol'] for s in simulated_stats]
    sim_skews = [s['skew'] for s in simulated_stats]
    sim_kurts = [s['kurt'] for s in simulated_stats]
    sim_max_dds = [s['max_dd'] for s in simulated_stats]
    
    # Compute medians
    medians = {
        'cagr': np.median(sim_cagrs),
        'vol': np.median(sim_vols),
        'skew': np.median(sim_skews),
        'kurt': np.median(sim_kurts),
        'max_dd': np.median(sim_max_dds)
    }
    
    # Compute percentiles for confidence intervals
    percentiles = {
        'cagr': (np.percentile(sim_cagrs, 5), np.percentile(sim_cagrs, 95)),
        'vol': (np.percentile(sim_vols, 5), np.percentile(sim_vols, 95)),
        'skew': (np.percentile(sim_skews, 5), np.percentile(sim_skews, 95)),
        'kurt': (np.percentile(sim_kurts, 5), np.percentile(sim_kurts, 95)),
        'max_dd': (np.percentile(sim_max_dds, 5), np.percentile(sim_max_dds, 95))
    }
    
    return {
        'empirical': emp_stats,
        'model_median': medians,
        'model_percentiles': percentiles,
        'all_simulations': simulated_stats
    }

def compute_per_regime_statistics(returns, regime_ids):
    """Compute statistics for each regime separately.
    
    Args:
        returns: Array of returns
        regime_ids: Array of regime IDs (same length as returns)
    
    Returns:
        dict: {regime_id: stats_dict} for each regime
    """
    regime_stats = {}
    
    # Normalize regime IDs to integers
    regime_ids_normalized = np.array([int(rid) for rid in regime_ids])
    
    for regime_id in np.unique(regime_ids_normalized):
        mask = regime_ids_normalized == regime_id
        regime_returns = returns[mask]
        
        if len(regime_returns) > 0:
            regime_stats[int(regime_id)] = compute_statistics(regime_returns)
    
    return regime_stats

def compare_per_regime(empirical_returns, empirical_regimes, simulated_paths, regime_ids):
    """Compare empirical vs model statistics per regime.
    
    Args:
        empirical_returns: Array of empirical returns
        empirical_regimes: Array of empirical regime IDs
        simulated_paths: List of (returns, regimes) tuples from simulations
        regime_ids: List of all regime IDs
    
    Returns:
        dict: {regime_id: comparison_dict} for each regime
    """
    # Normalize empirical regime IDs
    empirical_regimes_normalized = np.array([int(rid) for rid in empirical_regimes])
    
    # Compute empirical statistics per regime
    emp_regime_stats = compute_per_regime_statistics(empirical_returns, empirical_regimes_normalized)
    
    # Compute simulated statistics per regime
    sim_regime_stats = {rid: [] for rid in regime_ids}
    
    for returns, regimes in simulated_paths:
        # Normalize simulated regime IDs
        regimes_normalized = np.array([int(rid) for rid in regimes])
        sim_stats = compute_per_regime_statistics(returns, regimes_normalized)
        for rid, stats in sim_stats.items():
            if rid in sim_regime_stats:
                sim_regime_stats[rid].append(stats)
    
    # Compute medians and percentiles per regime
    regime_comparisons = {}
    
    for rid in regime_ids:
        if rid not in emp_regime_stats:
            continue
        
        emp_stats = emp_regime_stats[rid]
        sim_stats_list = sim_regime_stats.get(rid, [])
        
        if len(sim_stats_list) == 0:
            continue
        
        # Extract statistics
        sim_cagrs = [s['cagr'] for s in sim_stats_list]
        sim_vols = [s['vol'] for s in sim_stats_list]
        sim_skews = [s['skew'] for s in sim_stats_list]
        sim_kurts = [s['kurt'] for s in sim_stats_list]
        sim_max_dds = [s['max_dd'] for s in sim_stats_list]
        
        medians = {
            'cagr': np.median(sim_cagrs),
            'vol': np.median(sim_vols),
            'skew': np.median(sim_skews),
            'kurt': np.median(sim_kurts),
            'max_dd': np.median(sim_max_dds)
        }
        
        percentiles = {
            'cagr': (np.percentile(sim_cagrs, 5), np.percentile(sim_cagrs, 95)),
            'vol': (np.percentile(sim_vols, 5), np.percentile(sim_vols, 95)),
            'skew': (np.percentile(sim_skews, 5), np.percentile(sim_skews, 95)),
            'kurt': (np.percentile(sim_kurts, 5), np.percentile(sim_kurts, 95)),
            'max_dd': (np.percentile(sim_max_dds, 5), np.percentile(sim_max_dds, 95))
        }
        
        regime_comparisons[rid] = {
            'empirical': emp_stats,
            'model_median': medians,
            'model_percentiles': percentiles,
            'n_empirical': np.sum(empirical_regimes_normalized == rid),
            'n_simulations': len(sim_stats_list)
        }
    
    return regime_comparisons

def display_comparison_table(comparison_results, title="Empirical vs Model Statistics Comparison"):
    """Display comparison table of empirical vs model statistics."""
    emp = comparison_results['empirical']
    med = comparison_results['model_median']
    pct = comparison_results['model_percentiles']
    
    table = Table(title=title)
    table.add_column("Statistic", style="cyan", width=20)
    table.add_column("Empirical", justify="right", style="green", width=15)
    table.add_column("Model Median", justify="right", style="blue", width=15)
    table.add_column("5th Percentile", justify="right", style="dim", width=15)
    table.add_column("95th Percentile", justify="right", style="dim", width=15)
    table.add_column("Difference", justify="right", style="yellow", width=15)
    table.add_column("% Error", justify="right", style="red", width=12)
    
    stats_names = {
        'cagr': 'CAGR (annual)',
        'vol': 'Annual Volatility',
        'skew': 'Skewness',
        'kurt': 'Kurtosis (excess)',
        'max_dd': 'Max Drawdown'
    }
    
    for stat_key, stat_name in stats_names.items():
        emp_val = emp[stat_key]
        med_val = med[stat_key]
        pct_5 = pct[stat_key][0]
        pct_95 = pct[stat_key][1]
        diff = med_val - emp_val
        
        # Compute percentage error
        if abs(emp_val) > 1e-6:
            pct_error = (diff / abs(emp_val)) * 100
        else:
            pct_error = diff * 100 if abs(diff) > 1e-6 else 0.0
        
        # Format values
        if stat_key == 'cagr' or stat_key == 'vol':
            emp_str = f"{emp_val:.4f}"
            med_str = f"{med_val:.4f}"
            pct_5_str = f"{pct_5:.4f}"
            pct_95_str = f"{pct_95:.4f}"
            diff_str = f"{diff:.4f}"
        elif stat_key == 'max_dd':
            emp_str = f"{emp_val:.4f}"
            med_str = f"{med_val:.4f}"
            pct_5_str = f"{pct_5:.4f}"
            pct_95_str = f"{pct_95:.4f}"
            diff_str = f"{diff:.4f}"
        else:
            emp_str = f"{emp_val:.3f}"
            med_str = f"{med_val:.3f}"
            pct_5_str = f"{pct_5:.3f}"
            pct_95_str = f"{pct_95:.3f}"
            diff_str = f"{diff:.3f}"
        
        pct_error_str = f"{pct_error:.2f}%"
        
        # Color code based on error
        if abs(pct_error) < 5:
            error_style = "green"
        elif abs(pct_error) < 15:
            error_style = "yellow"
        else:
            error_style = "red"
        
        table.add_row(
            stat_name,
            emp_str,
            med_str,
            pct_5_str,
            pct_95_str,
            f"[{error_style}]{diff_str}[/{error_style}]",
            f"[{error_style}]{pct_error_str}[/{error_style}]"
        )
    
    console.print(table)

def display_per_regime_tables(regime_comparisons):
    """Display comparison tables for each regime."""
    for regime_id in sorted(regime_comparisons.keys()):
        comp = regime_comparisons[regime_id]
        title = f"Regime {regime_id} Comparison (Empirical n={comp['n_empirical']}, Simulations n={comp['n_simulations']})"
        display_comparison_table(comp, title=title)
        console.print()  # Blank line between regimes

def display_moment_matching_quality(moment_matching_info, regime_ids):
    """Display moment matching quality diagnostics from the fitting process.
    
    Args:
        moment_matching_info: dict with moment matching results per regime
        regime_ids: List of regime IDs
    """
    console.print("\n[bold cyan]Moment Matching Quality (from Parameter Fitting)[/bold cyan]")
    console.print("=" * 80)
    
    table = Table(title="Moment Matching Errors from Fitting Process")
    table.add_column("Regime", style="cyan", width=8)
    table.add_column("N Obs", justify="right", width=8)
    table.add_column("Mean Error %", justify="right", width=12)
    table.add_column("Std Error %", justify="right", width=12)
    table.add_column("Skew Error %", justify="right", width=12)
    table.add_column("Kurt Error %", justify="right", width=12)
    table.add_column("Max DD Error %", justify="right", width=15)
    table.add_column("Objective", justify="right", width=12)
    
    for regime_id in sorted(regime_ids):
        if regime_id not in moment_matching_info:
            continue
        
        info = moment_matching_info[regime_id]
        
        # Color code based on error magnitude
        def get_error_style(pct_error):
            if abs(pct_error) < 1:
                return "green"
            elif abs(pct_error) < 5:
                return "yellow"
            else:
                return "red"
        
        mean_style = get_error_style(info['mean_error_pct'])
        std_style = get_error_style(info['std_error_pct'])
        skew_style = get_error_style(info['skew_error_pct'])
        kurt_style = get_error_style(info['kurt_error_pct'])
        dd_style = get_error_style(info['max_dd_error_pct'])
        
        table.add_row(
            f"Regime {int(regime_id)}",
            str(info['n_obs']),
            f"[{mean_style}]{info['mean_error_pct']:.2f}%[/{mean_style}]",
            f"[{std_style}]{info['std_error_pct']:.2f}%[/{std_style}]",
            f"[{skew_style}]{info['skew_error_pct']:.2f}%[/{skew_style}]",
            f"[{kurt_style}]{info['kurt_error_pct']:.2f}%[/{kurt_style}]",
            f"[{dd_style}]{info['max_dd_error_pct']:.2f}%[/{dd_style}]",
            f"{info['objective']:.2e}"
        )
    
    console.print(table)
    
    # Check for potential issues
    console.print("\n[bold cyan]Moment Matching Quality Assessment:[/bold cyan]")
    issues = []
    
    for regime_id in sorted(regime_ids):
        if regime_id not in moment_matching_info:
            continue
        
        info = moment_matching_info[regime_id]
        
        # Check for large errors
        if abs(info['mean_error_pct']) > 5:
            issues.append(f"Regime {int(regime_id)}: Mean error {info['mean_error_pct']:.2f}% > 5%")
        if abs(info['std_error_pct']) > 5:
            issues.append(f"Regime {int(regime_id)}: Std error {info['std_error_pct']:.2f}% > 5%")
        if abs(info['skew_error_pct']) > 10:
            issues.append(f"Regime {int(regime_id)}: Skew error {info['skew_error_pct']:.2f}% > 10%")
        if abs(info['kurt_error_pct']) > 10:
            issues.append(f"Regime {int(regime_id)}: Kurt error {info['kurt_error_pct']:.2f}% > 10%")
        if abs(info['max_dd_error_pct']) > 20:
            issues.append(f"Regime {int(regime_id)}: Max DD error {info['max_dd_error_pct']:.2f}% > 20%")
    
    if issues:
        console.print("[yellow]⚠ Potential Issues Detected:[/yellow]")
        for issue in issues:
            console.print(f"  • {issue}")
        console.print("\n[dim]Note: Large moment matching errors may indicate parameter estimation issues.[/dim]")
        console.print("[dim]Consider re-running moment matching with different settings or bounds.[/dim]")
    else:
        console.print("[green]✓ Moment matching quality looks good (all errors within acceptable ranges)[/green]")
    
    console.print()

def display_parameter_estimates(params_dict, regime_ids):
    """Display MJD model parameter estimates for each regime.
    
    Args:
        params_dict: {regime_id: params_array} for each regime
        regime_ids: List of regime IDs
    """
    table = Table(title="Merton Jump Diffusion Model Parameter Estimates by Regime")
    table.add_column("Regime", style="cyan", width=8)
    table.add_column("μ (annual)", justify="right", style="green", width=12)
    table.add_column("σ (annual)", justify="right", width=12)
    table.add_column("λ (annual)", justify="right", width=12)
    table.add_column("μ_J", justify="right", width=10)
    table.add_column("σ_J", justify="right", width=10)
    
    param_names = ['mu', 'sigma', 'lam', 'mu_J', 'sigma_J']
    
    for regime_id in sorted(regime_ids):
        if regime_id not in params_dict:
            continue
        
        params = params_dict[regime_id]
        row_data = [f"Regime {regime_id}"]
        
        for param_name in param_names:
            param_idx = IDX[param_name]
            param_val = params[param_idx]
            
            # Format based on parameter type
            if param_name in ['mu', 'sigma', 'mu_J', 'sigma_J']:
                row_data.append(f"{param_val:.4f}")
            elif param_name == 'lam':
                row_data.append(f"{param_val:.3f}")
            else:
                row_data.append(f"{param_val:.4f}")
        
        table.add_row(*row_data)
    
    console.print(table)
    
    # Also display parameter interpretation
    console.print("\n[bold cyan]Parameter Interpretation:[/bold cyan]")
    console.print("  μ (mu): Annual drift rate")
    console.print("  σ (sigma): Annual volatility (constant)")
    console.print("  λ (lambda): Jump intensity per year")
    console.print("  μ_J: Mean jump size")
    console.print("  σ_J: Jump size volatility")
    console.print()
    
    # Parameter comparison analysis
    if len(params_dict) > 1:
        console.print("[bold cyan]Parameter Comparison Across Regimes:[/bold cyan]")
        console.print("=" * 80)
        
        param_names = ['mu', 'sigma', 'lam', 'mu_J', 'sigma_J']
        param_labels = {
            'mu': 'Drift (μ)',
            'sigma': 'Volatility (σ)',
            'lam': 'Jump Intensity (λ)',
            'mu_J': 'Jump Mean (μ_J)',
            'sigma_J': 'Jump Vol (σ_J)'
        }
        
        for param_name in param_names:
            param_idx = IDX[param_name]
            values = [params_dict[rid][param_idx] for rid in sorted(regime_ids) if rid in params_dict]
            regimes_list = [rid for rid in sorted(regime_ids) if rid in params_dict]
            
            if len(values) > 1:
                min_val = min(values)
                max_val = max(values)
                min_regime = regimes_list[values.index(min_val)]
                max_regime = regimes_list[values.index(max_val)]
                range_val = max_val - min_val
                
                # Format based on parameter type
                if param_name in ['mu', 'sigma', 'mu_J', 'sigma_J']:
                    fmt = ".4f"
                else:
                    fmt = ".3f"
                
                console.print(f"  {param_labels[param_name]}:")
                console.print(f"    Range: [{min_val:{fmt}}, {max_val:{fmt}}] (span: {range_val:{fmt}})")
                console.print(f"    Min: Regime {min_regime} = {min_val:{fmt}}")
                console.print(f"    Max: Regime {max_regime} = {max_val:{fmt}}")
        
        console.print()

# ============================================================================
# PLOTTING
# ============================================================================

def plot_historical_vs_simulations(empirical_returns, simulated_paths, output_path, n_sample_paths=50):
    """Plot historical returns vs simulated paths.
    
    Args:
        empirical_returns: Array of empirical returns
        simulated_paths: List of (returns, regimes) tuples
        output_path: Path to save the plot
        n_sample_paths: Number of simulation paths to plot (for clarity)
    """
    # Set dark background style
    plt.style.use('dark_background')
    
    fig = plt.figure(figsize=(16, 10), facecolor='black')
    gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.3, height_ratios=[2, 1, 1])
    
    # Convert returns to cumulative prices
    empirical_prices = np.cumprod(1.0 + empirical_returns) * 100  # Start at 100
    
    # Plot 1: Cumulative price paths
    ax1 = fig.add_subplot(gs[0])
    
    # Plot historical
    time_empirical = np.arange(len(empirical_prices)) / PERIODS_PER_YEAR
    ax1.plot(time_empirical, empirical_prices, linewidth=2.5, color='cyan', 
             label='Historical', alpha=0.9, zorder=10)
    
    # Plot sample of simulated paths
    n_paths_to_plot = min(n_sample_paths, len(simulated_paths))
    indices = np.linspace(0, len(simulated_paths) - 1, n_paths_to_plot, dtype=int)
    
    for idx in indices:
        returns, _ = simulated_paths[idx]
        prices = np.cumprod(1.0 + returns) * 100
        time_sim = np.arange(len(prices)) / PERIODS_PER_YEAR
        ax1.plot(time_sim, prices, linewidth=0.8, color='orange', alpha=0.15, zorder=1)
    
    # Plot median and percentiles of all simulations
    all_prices = []
    min_length = min(len(empirical_returns), min(len(r) for r, _ in simulated_paths))
    
    for returns, _ in simulated_paths:
        prices = np.cumprod(1.0 + returns[:min_length]) * 100
        all_prices.append(prices)
    
    all_prices = np.array(all_prices)
    median_prices = np.median(all_prices, axis=0)
    pct_5_prices = np.percentile(all_prices, 5, axis=0)
    pct_95_prices = np.percentile(all_prices, 95, axis=0)
    
    time_median = np.arange(len(median_prices)) / PERIODS_PER_YEAR
    ax1.plot(time_median, median_prices, linewidth=2, color='yellow', 
             label='Model Median', linestyle='--', alpha=0.8, zorder=5)
    ax1.fill_between(time_median, pct_5_prices, pct_95_prices, 
                     color='orange', alpha=0.2, label='Model 5th-95th Percentile', zorder=2)
    
    ax1.set_xlabel('Years', fontsize=12, color='white')
    ax1.set_ylabel('Cumulative Price (Index = 100)', fontsize=12, color='white')
    ax1.set_title('Historical vs Simulated Price Paths', fontsize=14, fontweight='bold', color='white')
    ax1.legend(fontsize=11, facecolor='black', edgecolor='white', labelcolor='white', loc='upper left')
    ax1.grid(True, alpha=0.3, color='gray')
    ax1.set_facecolor('black')
    ax1.tick_params(colors='white')
    
    # Plot 2: Returns distribution comparison
    ax2 = fig.add_subplot(gs[1])
    
    # Flatten all simulated returns
    all_sim_returns = np.concatenate([r for r, _ in simulated_paths])
    
    # Histogram comparison
    bins = np.linspace(min(np.min(empirical_returns), np.min(all_sim_returns)),
                      max(np.max(empirical_returns), np.max(all_sim_returns)), 50)
    
    ax2.hist(empirical_returns, bins=bins, alpha=0.6, label='Historical', 
            density=True, color='cyan', edgecolor='white', linewidth=0.5)
    ax2.hist(all_sim_returns, bins=bins, alpha=0.6, label='Simulated (all paths)', 
            density=True, color='orange', edgecolor='white', linewidth=0.5)
    
    ax2.set_xlabel('Log Return', fontsize=12, color='white')
    ax2.set_ylabel('Density', fontsize=12, color='white')
    ax2.set_title('Returns Distribution Comparison', fontsize=13, fontweight='bold', color='white')
    ax2.legend(fontsize=11, facecolor='black', edgecolor='white', labelcolor='white')
    ax2.grid(True, alpha=0.3, color='gray')
    ax2.set_facecolor('black')
    ax2.tick_params(colors='white')
    
    # Plot 3: Drawdown comparison
    ax3 = fig.add_subplot(gs[2])
    
    # Compute drawdowns for historical
    emp_peak = np.maximum.accumulate(empirical_prices)
    emp_dd = (empirical_prices - emp_peak) / emp_peak * 100
    time_emp_dd = np.arange(len(emp_dd)) / PERIODS_PER_YEAR
    
    ax3.plot(time_emp_dd, emp_dd, linewidth=2, color='cyan', label='Historical Drawdown', alpha=0.9)
    
    # Compute drawdowns for median simulation
    sim_peak = np.maximum.accumulate(median_prices)
    sim_dd = (median_prices - sim_peak) / sim_peak * 100
    
    ax3.plot(time_median, sim_dd, linewidth=2, color='yellow', 
             label='Model Median Drawdown', linestyle='--', alpha=0.8)
    
    # Fill percentile range
    sim_pct_5_peak = np.maximum.accumulate(pct_5_prices)
    sim_pct_5_dd = (pct_5_prices - sim_pct_5_peak) / sim_pct_5_peak * 100
    sim_pct_95_peak = np.maximum.accumulate(pct_95_prices)
    sim_pct_95_dd = (pct_95_prices - sim_pct_95_peak) / sim_pct_95_peak * 100
    
    ax3.fill_between(time_median, sim_pct_5_dd, sim_pct_95_dd, 
                     color='orange', alpha=0.2, label='Model 5th-95th Percentile')
    
    ax3.set_xlabel('Years', fontsize=12, color='white')
    ax3.set_ylabel('Drawdown (%)', fontsize=12, color='white')
    ax3.set_title('Drawdown Comparison', fontsize=13, fontweight='bold', color='white')
    ax3.legend(fontsize=11, facecolor='black', edgecolor='white', labelcolor='white', loc='lower left')
    ax3.grid(True, alpha=0.3, color='gray')
    ax3.set_facecolor('black')
    ax3.tick_params(colors='white')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='black')
    plt.close()
    
    console.print(f"[green]✓ Plot saved to: {output_path}[/green]")

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main function to run Monte Carlo simulation and comparison."""
    parser = argparse.ArgumentParser(description='Monte Carlo simulation with regime-switching Merton Jump Diffusion model')
    parser.add_argument('--n-simulations', type=int, default=1000,
                       help='Number of Monte Carlo simulations (default: 1000)')
    parser.add_argument('--n-periods', type=int, default=None,
                       help='Number of periods per simulation (default: match empirical data length)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--mjd-csv', type=str, 
                       default='data/MJD_per_regime_Moment_Matching_results.csv',
                       help='Path to MJD parameters CSV')
    parser.add_argument('--transition-csv', type=str,
                       default='data/HMM_transition_matrix.csv',
                       help='Path to HMM transition matrix CSV')
    parser.add_argument('--returns-csv', type=str,
                       default='data/regime_classification_nominal_returns.csv',
                       help='Path to empirical returns CSV')
    parser.add_argument('--plot', action='store_true',
                       help='Generate and save comparison plots')
    parser.add_argument('--n-sample-paths', type=int, default=50,
                       help='Number of simulation paths to plot (default: 50)')
    args = parser.parse_args()
    
    console.print("[bold cyan]Monte Carlo Simulation: Regime-Switching Merton Jump Diffusion Model[/bold cyan]")
    console.print("=" * 80)
    
    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    grandparent_dir = os.path.dirname(parent_dir)
    
    mjd_path = args.mjd_csv
    if not os.path.exists(mjd_path):
        mjd_path = os.path.join(grandparent_dir, args.mjd_csv)
    
    transition_path = args.transition_csv
    if not os.path.exists(transition_path):
        transition_path = os.path.join(grandparent_dir, args.transition_csv)
    
    returns_path = args.returns_csv
    if not os.path.exists(returns_path):
        returns_path = os.path.join(grandparent_dir, args.returns_csv)
    
    # Load data
    console.print("\n[cyan]Loading data...[/cyan]")
    console.print(f"  MJD parameters: {mjd_path}")
    console.print(f"  Transition matrix: {transition_path}")
    console.print(f"  Empirical returns: {returns_path}")
    
    params_dict, moment_matching_info = load_mjd_parameters(mjd_path)
    transition_matrix, regime_ids = load_transition_matrix(transition_path)
    empirical_returns, empirical_regimes = load_empirical_data(returns_path)
    
    # Normalize regime IDs to integers for consistency
    params_dict_normalized = {}
    for regime_id, params in params_dict.items():
        normalized_id = int(regime_id)
        params_dict_normalized[normalized_id] = params
    
    regime_ids_normalized = [int(rid) for rid in regime_ids]
    params_dict = params_dict_normalized
    regime_ids = regime_ids_normalized
    
    # Validate that all regimes in params_dict are in transition matrix
    missing_regimes = set(params_dict.keys()) - set(regime_ids)
    if missing_regimes:
        console.print(f"[red]Error: Regimes {missing_regimes} in parameters but not in transition matrix[/red]")
        return
    
    console.print(f"[green]✓ Loaded {len(params_dict)} regimes[/green]")
    console.print(f"[green]✓ Transition matrix: {transition_matrix.shape}[/green]")
    console.print(f"[green]✓ Empirical data: {len(empirical_returns)} periods[/green]")
    console.print(f"[dim]  Regime IDs: {sorted(regime_ids)}[/dim]")
    
    # Display moment matching quality first (to check if parameters are well-fitted)
    display_moment_matching_quality(moment_matching_info, regime_ids)
    
    # Display parameter estimates
    console.print("\n[bold cyan]Merton Jump Diffusion Model Parameter Estimates[/bold cyan]")
    console.print("=" * 80)
    display_parameter_estimates(params_dict, regime_ids)
    
    # Display transition matrix
    console.print("\n[bold cyan]HMM Transition Matrix[/bold cyan]")
    console.print("=" * 80)
    trans_table = Table(title="Transition Probabilities")
    trans_table.add_column("From → To", style="cyan", width=15)
    for rid in regime_ids:
        trans_table.add_column(f"Regime {rid}", justify="right", width=12)
    
    for i, from_rid in enumerate(regime_ids):
        row_data = [f"Regime {from_rid}"]
        for j, to_rid in enumerate(regime_ids):
            prob = transition_matrix[i, j]
            row_data.append(f"{prob:.4f}")
        trans_table.add_row(*row_data)
    
    console.print(trans_table)
    console.print()
    
    # Set n_periods if not specified
    n_periods = args.n_periods if args.n_periods is not None else len(empirical_returns)
    
    # Remind user about plot flag
    if not args.plot:
        console.print("[dim]💡 Tip: Use --plot flag to generate comparison plots[/dim]")
        console.print()
    
    # Run Monte Carlo simulation
    console.print(f"\n[cyan]Running {args.n_simulations} Monte Carlo simulations...[/cyan]")
    console.print(f"  Periods per simulation: {n_periods}")
    console.print(f"  Random seed: {args.seed}")
    
    # Get paths if plotting or per-regime analysis is needed
    return_paths = args.plot or True  # Always return paths for per-regime analysis
    result = run_monte_carlo_simulation(
        params_dict, transition_matrix, regime_ids,
        n_simulations=args.n_simulations,
        n_periods=n_periods,
        seed=args.seed,
        return_paths=return_paths
    )
    
    if return_paths:
        simulated_stats, simulated_paths = result
    else:
        simulated_stats = result
        simulated_paths = None
    
    # Compare results (overall)
    console.print("\n[cyan]Comparing empirical vs model statistics (overall)...[/cyan]")
    comparison_results = compare_empirical_vs_model(empirical_returns, simulated_stats)
    
    # Display overall results
    console.print("\n")
    display_comparison_table(comparison_results)
    
    # Per-regime comparison
    if empirical_regimes is not None and simulated_paths is not None and len(empirical_regimes) == len(empirical_returns):
        console.print("\n[cyan]Comparing empirical vs model statistics (per regime)...[/cyan]")
        regime_comparisons = compare_per_regime(
            empirical_returns, empirical_regimes, simulated_paths, regime_ids
        )
        
        if regime_comparisons:
            console.print("\n")
            display_per_regime_tables(regime_comparisons)
            
            # Compare moment matching predictions vs simulation results
            console.print("\n[bold cyan]Moment Matching Predictions vs Simulation Results[/bold cyan]")
            console.print("=" * 80)
            console.print("[dim]Comparing what moment matching predicted vs what we get in full simulations[/dim]")
            console.print()
            
            diagnostic_table = Table(title="Moment Matching Fit vs Simulation Reality")
            diagnostic_table.add_column("Regime", style="cyan", width=8)
            diagnostic_table.add_column("Metric", style="yellow", width=20)
            diagnostic_table.add_column("Moment Match Predicted", justify="right", style="blue", width=18)
            diagnostic_table.add_column("Simulation Median", justify="right", style="green", width=18)
            diagnostic_table.add_column("Difference", justify="right", style="red", width=15)
            
            for regime_id in sorted(regime_comparisons.keys()):
                if regime_id not in moment_matching_info:
                    continue
                
                mm_info = moment_matching_info[regime_id]
                sim_comp = regime_comparisons[regime_id]
                
                # Compare CAGR (annualized mean)
                # Moment matching CSV has monthly mean, convert to annual
                mm_cagr = mm_info['model_mean'] * PERIODS_PER_YEAR
                sim_cagr = sim_comp['model_median']['cagr']
                cagr_diff = sim_cagr - mm_cagr
                cagr_pct_diff = (cagr_diff / abs(mm_cagr) * 100) if abs(mm_cagr) > 1e-6 else 0.0
                diagnostic_table.add_row(
                    f"Regime {regime_id}",
                    "CAGR (annual)",
                    f"{mm_cagr:.4f}",
                    f"{sim_cagr:.4f}",
                    f"{cagr_diff:.4f} ({cagr_pct_diff:+.1f}%)"
                )
                
                # Compare Volatility
                # Moment matching CSV has monthly std, convert to annual
                mm_vol = mm_info['model_std'] * math.sqrt(PERIODS_PER_YEAR)
                sim_vol = sim_comp['model_median']['vol']
                vol_diff = sim_vol - mm_vol
                vol_pct_diff = (vol_diff / abs(mm_vol) * 100) if abs(mm_vol) > 1e-6 else 0.0
                diagnostic_table.add_row(
                    "",
                    "Volatility (annual)",
                    f"{mm_vol:.4f}",
                    f"{sim_vol:.4f}",
                    f"{vol_diff:.4f} ({vol_pct_diff:+.1f}%)"
                )
                
                # Compare Skewness (same for monthly/annual)
                mm_skew = mm_info['model_skew']
                sim_skew = sim_comp['model_median']['skew']
                skew_diff = sim_skew - mm_skew
                skew_pct_diff = (skew_diff / abs(mm_skew) * 100) if abs(mm_skew) > 1e-6 else 0.0
                diagnostic_table.add_row(
                    "",
                    "Skewness",
                    f"{mm_skew:.3f}",
                    f"{sim_skew:.3f}",
                    f"{skew_diff:.3f} ({skew_pct_diff:+.1f}%)"
                )
                
                # Compare Kurtosis (same for monthly/annual)
                mm_kurt = mm_info['model_kurt']
                sim_kurt = sim_comp['model_median']['kurt']
                kurt_diff = sim_kurt - mm_kurt
                kurt_pct_diff = (kurt_diff / abs(mm_kurt) * 100) if abs(mm_kurt) > 1e-6 else 0.0
                diagnostic_table.add_row(
                    "",
                    "Kurtosis (excess)",
                    f"{mm_kurt:.3f}",
                    f"{sim_kurt:.3f}",
                    f"{kurt_diff:.3f} ({kurt_pct_diff:+.1f}%)"
                )
                
                # Compare Max Drawdown (same for monthly/annual)
                mm_dd = mm_info['model_max_dd']
                sim_dd = sim_comp['model_median']['max_dd']
                dd_diff = sim_dd - mm_dd
                dd_pct_diff = (dd_diff / abs(mm_dd) * 100) if abs(mm_dd) > 1e-6 else 0.0
                diagnostic_table.add_row(
                    "",
                    "Max Drawdown",
                    f"{mm_dd:.4f}",
                    f"{sim_dd:.4f}",
                    f"{dd_diff:.4f} ({dd_pct_diff:+.1f}%)"
                )
                
                diagnostic_table.add_row("", "", "", "", "")  # Spacer
            
            console.print(diagnostic_table)
            console.print("\n[dim]Note: Large differences indicate that regime-switching behavior may differ from[/dim]")
            console.print("[dim]isolated regime behavior, or that moment matching analytical formulas may not[/dim]")
            console.print("[dim]perfectly match simulated behavior in the full regime-switching context.[/dim]")
            console.print()
            
            # Additional diagnostic: Test isolated regime simulations
            console.print("\n[bold cyan]Isolated Regime Test (No Switching)[/bold cyan]")
            console.print("=" * 80)
            console.print("[dim]Simulating each regime in isolation to see if parameters match expectations[/dim]")
            console.print()
            
            isolated_test_table = Table(title="Isolated Regime Simulations (2 paths × 500 periods = 1000 total, with 100-period burn-in)")
            isolated_test_table.add_column("Regime", style="cyan", width=8)
            isolated_test_table.add_column("Metric", style="yellow", width=20)
            isolated_test_table.add_column("Moment Match Predicted", justify="right", style="blue", width=18)
            isolated_test_table.add_column("Isolated Simulation", justify="right", style="green", width=18)
            isolated_test_table.add_column("Difference", justify="right", style="red", width=15)
            
            for regime_id in sorted(regime_comparisons.keys()):
                if regime_id not in moment_matching_info or regime_id not in params_dict:
                    continue
                
                # Simulate this regime in isolation (no switching)
                params = params_dict[regime_id]
                mm_info = moment_matching_info[regime_id]
                
                # Run isolated simulation with burn-in
                # MJD doesn't need equilibrium variance, but we keep burn-in for consistency
                isolated_returns = []
                burn_in = 100
                path_length = 500  # Longer paths for better statistics
                
                for path_idx in range(2):  # 2 paths, 500 periods each = 1000 total periods
                    returns, _ = simulate_regime_switching_mjd(
                        {regime_id: params},  # Only one regime
                        np.array([[1.0]]),  # Stay in same regime (100% probability)
                        [regime_id],
                        n_periods=path_length,
                        initial_regime=regime_id,
                        seed=path_idx,  # Different seed per path
                        burn_in_periods=burn_in
                    )
                    isolated_returns.extend(returns)
                
                isolated_returns = np.array(isolated_returns)
                isolated_stats = compute_statistics(isolated_returns)
                isolated_stats_exp = isolated_stats  # MJD doesn't have variance distinction
                
                # Compare to moment matching predictions
                mm_cagr = mm_info['model_mean'] * PERIODS_PER_YEAR
                iso_cagr = isolated_stats['cagr']
                cagr_diff = iso_cagr - mm_cagr
                cagr_pct = (cagr_diff / abs(mm_cagr) * 100) if abs(mm_cagr) > 1e-6 else 0.0
                
                isolated_test_table.add_row(
                    f"Regime {regime_id}",
                    "CAGR (annual)",
                    f"{mm_cagr:.4f}",
                    f"{iso_cagr:.4f}",
                    f"{cagr_diff:.4f} ({cagr_pct:+.1f}%)"
                )
                
                mm_vol = mm_info['model_std'] * math.sqrt(PERIODS_PER_YEAR)
                iso_vol = isolated_stats['vol']
                vol_diff = iso_vol - mm_vol
                vol_pct = (vol_diff / abs(mm_vol) * 100) if abs(mm_vol) > 1e-6 else 0.0
                
                isolated_test_table.add_row(
                    "",
                    "Volatility (annual)",
                    f"{mm_vol:.4f}",
                    f"{iso_vol:.4f}",
                    f"{vol_diff:.4f} ({vol_pct:+.1f}%)"
                )
                
                mm_skew = mm_info['model_skew']
                iso_skew = isolated_stats['skew']
                skew_diff = iso_skew - mm_skew
                skew_pct = (skew_diff / abs(mm_skew) * 100) if abs(mm_skew) > 1e-6 else 0.0
                
                isolated_test_table.add_row(
                    "",
                    "Skewness",
                    f"{mm_skew:.3f}",
                    f"{iso_skew:.3f}",
                    f"{skew_diff:.3f} ({skew_pct:+.1f}%)"
                )
                
                isolated_test_table.add_row("", "", "", "", "")  # Spacer
            
            console.print(isolated_test_table)
            console.print("\n[dim]Note: Isolated simulations use 100-period burn-in for consistency.[/dim]")
            console.print("[dim]MJD model uses constant volatility, so no variance equilibrium is needed.[/dim]")
            console.print("\n[dim]If isolated simulations match moment matching but regime-switching doesn't,[/dim]")
            console.print("[dim]the issue is likely transition probabilities or regime-switching effects.[/dim]")
            console.print()
            
            # Detailed analysis of analytical formula issues
            console.print("\n[bold cyan]Analytical Formula Validation[/bold cyan]")
            console.print("=" * 80)
            
            # Check which formulas are problematic
            formula_issues = []
            for regime_id in sorted(regime_comparisons.keys()):
                if regime_id not in moment_matching_info or regime_id not in params_dict:
                    continue
                
                mm_info = moment_matching_info[regime_id]
                params = params_dict[regime_id]
                
                # Get isolated simulation results with burn-in
                isolated_returns = []
                burn_in = 100
                path_length = 500
                
                for path_idx in range(2):
                    returns, _ = simulate_regime_switching_mjd(
                        {regime_id: params},
                        np.array([[1.0]]),
                        [regime_id],
                        n_periods=path_length,
                        initial_regime=regime_id,
                        seed=path_idx,
                        burn_in_periods=burn_in
                    )
                    isolated_returns.extend(returns)
                
                isolated_returns = np.array(isolated_returns)
                isolated_stats = compute_statistics(isolated_returns)
                
                # Compare mean
                mm_mean = mm_info['model_mean']
                iso_mean = isolated_stats['cagr'] / PERIODS_PER_YEAR  # Convert to monthly
                mean_error = abs(iso_mean - mm_mean) / abs(mm_mean) * 100 if abs(mm_mean) > 1e-6 else 0.0
                
                # Compare std (should be exact for MJD)
                mm_std = mm_info['model_std']
                iso_std = isolated_stats['vol'] / math.sqrt(PERIODS_PER_YEAR)  # Convert to monthly
                std_error = abs(iso_std - mm_std) / abs(mm_std) * 100 if abs(mm_std) > 1e-6 else 0.0
                
                # Compare skew (exact formula for MJD)
                mm_skew = mm_info['model_skew']
                iso_skew = isolated_stats['skew']
                skew_error = abs(iso_skew - mm_skew) / abs(mm_skew) * 100 if abs(mm_skew) > 1e-6 else abs(iso_skew - mm_skew)
                
                # Compare kurt (exact formula for MJD)
                mm_kurt = mm_info['model_kurt']
                iso_kurt = isolated_stats['kurt']
                kurt_error = abs(iso_kurt - mm_kurt) / abs(mm_kurt) * 100 if abs(mm_kurt) > 1e-6 else abs(iso_kurt - mm_kurt)
                
                if mean_error > 5:
                    formula_issues.append(f"Regime {regime_id}: Mean formula error {mean_error:.1f}% (should be exact!)")
                if std_error > 5:
                    formula_issues.append(f"Regime {regime_id}: Std formula error {std_error:.1f}% (should be exact!)")
                if abs(skew_error) > 50:
                    formula_issues.append(f"Regime {regime_id}: Skew formula error {skew_error:.1f}% (exact for MJD)")
                if abs(kurt_error) > 50:
                    formula_issues.append(f"Regime {regime_id}: Kurt formula error {kurt_error:.1f}% (exact for MJD)")
            
            if formula_issues:
                console.print("[red]⚠ Analytical Formula Issues Detected:[/red]")
                for issue in formula_issues:
                    console.print(f"  • {issue}")
                console.print()
                console.print("[yellow]The analytical formulas used in moment matching do not accurately predict[/yellow]")
                console.print("[yellow]simulated behavior. This explains why parameters fit well but simulations differ.[/yellow]")
                console.print()
                console.print("[dim]Note: MJD formulas are exact (no approximations), so errors may indicate implementation issues.[/dim]")
            else:
                console.print("[green]✓ Analytical formulas appear to match simulations reasonably well[/green]")
                console.print("[dim]  (MJD formulas are exact, so good match is expected)[/dim]")
            
            console.print()
    elif empirical_regimes is None:
        console.print("\n[yellow]⚠ Per-regime comparison skipped: No regime IDs in empirical data[/yellow]")
    
    # Summary and Root Cause Analysis
    console.print("\n[bold cyan]Summary (Overall)[/bold cyan]")
    console.print("=" * 80)
    emp = comparison_results['empirical']
    med = comparison_results['model_median']
    
    console.print(f"Empirical CAGR: {emp['cagr']:.4f} | Model Median: {med['cagr']:.4f}")
    console.print(f"Empirical Volatility: {emp['vol']:.4f} | Model Median: {med['vol']:.4f}")
    console.print(f"Empirical Skewness: {emp['skew']:.3f} | Model Median: {med['skew']:.3f}")
    console.print(f"Empirical Kurtosis: {emp['kurt']:.3f} | Model Median: {med['kurt']:.3f}")
    console.print(f"Empirical Max DD: {emp['max_dd']:.4f} | Model Median: {med['max_dd']:.4f}")
    
    # Root cause analysis
    console.print("\n[bold cyan]Root Cause Analysis[/bold cyan]")
    console.print("=" * 80)
    
    # Check if moment matching quality is good but simulations differ
    mm_quality_good = True
    for regime_id in sorted(regime_ids):
        if regime_id in moment_matching_info:
            mm_info = moment_matching_info[regime_id]
            if (abs(mm_info['mean_error_pct']) > 5 or 
                abs(mm_info['std_error_pct']) > 5 or
                abs(mm_info['skew_error_pct']) > 10 or
                abs(mm_info['kurt_error_pct']) > 10):
                mm_quality_good = False
                break
    
    if mm_quality_good:
        console.print("[green]✓ Moment matching quality is good (parameters fit well during estimation)[/green]")
        console.print()
        console.print("[yellow]⚠ However, large discrepancies exist between predicted and simulated moments.[/yellow]")
        console.print()
        console.print("[bold]Likely Root Causes:[/bold]")
        console.print("1. [cyan]Regime-Switching Effects:[/cyan] MJD formulas are exact for single regimes,")
        console.print("   but regime-switching may introduce path-dependent effects not captured by")
        console.print("   isolated regime formulas.")
        console.print()
        console.print("2. [cyan]Regime Duration Effects:[/cyan] Analytical formulas assume infinite horizon,")
        console.print("   but actual regimes have finite durations, affecting convergence.")
        console.print()
        console.print("3. [cyan]Parameter Estimation Issues:[/cyan] Moment matching may not perfectly capture")
        console.print("   all distribution characteristics, especially tail behavior.")
        console.print()
        console.print("[bold]Potential Solutions:[/bold]")
        console.print()
        console.print("[bold cyan]Option 1 (RECOMMENDED):[/bold cyan] Use simulation-based moment matching")
        console.print("  Instead of analytical formulas, use Monte Carlo simulations during parameter fitting.")
        console.print("  This ensures parameters match actual simulated behavior in regime-switching context.")
        console.print()
        console.print("[bold cyan]Option 2:[/bold cyan] Improve parameter bounds or optimization")
        console.print("  MJD formulas are exact, so discrepancies may indicate optimization issues.")
        console.print("  Consider:")
        console.print("  - More optimization restarts")
        console.print("  - Better initial parameter guesses")
        console.print("  - Tighter parameter bounds")
        console.print()
        console.print("Option 3: Adjust parameters post-hoc")
        console.print("  Calibrate parameters to match regime-switching simulation results rather than")
        console.print("  isolated regime analytical formulas.")
        console.print()
        console.print("Option 4: Use hybrid approach")
        console.print("  Fit to analytical formulas first, then fine-tune using simulation-based optimization")
        console.print("  to account for regime-switching effects.")
        console.print()
        console.print("[bold yellow]Immediate Action Items:[/bold yellow]")
        console.print("1. Review moment matching results - MJD formulas are exact, so errors should be small")
        console.print("2. Consider re-fitting with simulation-based moment matching if discrepancies persist")
        console.print("3. Check if transition probabilities are correctly specified")
        console.print("4. Verify parameter loading from CSV matches expected format")
        console.print()
    else:
        console.print("[red]✗ Moment matching quality is poor - parameters may not be well-estimated[/red]")
        console.print("[dim]Consider re-running moment matching with:[/dim]")
        console.print("  • More optimization restarts")
        console.print("  • Different parameter bounds")
        console.print("  • Higher precision tolerances")
        console.print("  • --match-max-dd flag for better tail risk matching")
        console.print("  • Note: MJD formulas are exact, so good fit should be achievable")
        console.print()
    
    # Generate plot if requested
    if args.plot and simulated_paths is not None:
        console.print("\n[cyan]Generating comparison plot...[/cyan]")
        os.makedirs('output', exist_ok=True)
        output_path = os.path.join('output', 'historical_vs_simulations.png')
        plot_historical_vs_simulations(
            empirical_returns, simulated_paths, output_path, 
            n_sample_paths=args.n_sample_paths
        )
    elif not args.plot:
        console.print("\n[yellow]⚠ Plots not generated. Use --plot flag to create comparison plots.[/yellow]")
    
    console.print(f"\n[bold green]✓ Simulation complete![/bold green]")

if __name__ == "__main__":
    main()

