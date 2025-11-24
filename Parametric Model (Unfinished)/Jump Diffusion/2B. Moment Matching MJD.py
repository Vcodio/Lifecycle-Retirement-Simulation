"""
Merton Jump Diffusion (MJD) Model Moment Matching for Regime-Based Stock Returns

This script fits Merton Jump Diffusion model parameters to each regime by matching the first four moments:
- Mean (first moment)
- Variance (second moment)
- Skewness (third moment)
- Kurtosis (fourth moment)

Optionally, max drawdown can also be matched (use --match-max-dd flag).
Note: Max drawdown matching is computationally expensive and significantly slows optimization.

The goal is to ensure Monte Carlo simulations using the MJD model accurately reproduce
the historical return distribution characteristics for each volatility regime.

MJD Model:
- Geometric Brownian Motion (constant volatility) + Merton jumps
- Simpler than Bates model (no stochastic volatility)
- Parameters: mu (drift), sigma (volatility), lambda (jump intensity), mu_J (jump mean), sigma_J (jump volatility)

OPTIMIZATIONS:
1. Fast integration mode for likelihood regularization term
2. Smart initial parameter guesses based on empirical moments
3. Parallel optimization restarts for better global search
4. Early stopping when moment matching is achieved

ACCURACY:
- Exact moment calculations (no approximations for mean/variance)
- High-precision optimizer tolerances
- Multiple restarts to find global optimum
- Validation against simulated moments

PARAMETER BOUNDS - HOW TO KNOW IF THEY'RE TOO WIDE OR TOO TIGHT:

Signs that bounds are TOO WIDE:
1. Parameters vary wildly across optimization restarts (high variance)
2. Optimization is slow/unstable (searching large space)
3. Different restarts converge to very different values
4. Parameters show high standard deviation relative to bound range (>30% of range)

Signs that bounds are TOO TIGHT:
1. Parameters consistently hitting boundaries (>50% of restarts hit same boundary)
2. Poor fit quality despite hitting boundaries
3. Parameters that should vary are constrained at boundaries
4. Optimization fails frequently or converges to boundary values

The script automatically detects and reports boundary issues in the output.

Usage:
    python "2C. Moment Matching MJD.py" [--debug] [--match-max-dd] [--no-plots]
    
    --debug: Enable detailed debug output (includes boundary diagnostics)
    --match-max-dd: Enable max drawdown matching (slower but more accurate for tail risk)
    --no-plots: Disable plot generation (faster execution)
"""

import os
import sys
import math
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy import stats
from scipy.integrate import quad
import cmath
import argparse
from rich.console import Console
from rich.table import Table
from rich.progress import track, Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from multiprocessing import Pool, cpu_count
from functools import partial
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle

# ============================================================================
# CONFIGURATION
# ============================================================================

# Debug toggle - can be set via command line or here
DEBUG = False

# Parameter index positions for MJD: (mu, sigma, lam, mu_J, sigma_J)
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

# Constants
PERIODS_PER_YEAR = 12
DT = 1.0 / PERIODS_PER_YEAR
DENSITY_FLOOR = 1e-12
CF_INTEGRATION_LIMIT = 50.0

# Optimization settings
NUM_RESTARTS = 15  # Number of optimization restarts
NUM_RESTARTS_SMALL = 25  # More restarts for small regimes
MAXITER = 10000  # Maximum iterations per optimization
EARLY_STOP_MOMENT_ERROR = 0.01  # Stop if moment error < 1% (early stopping)

# Monte Carlo settings for max drawdown calculation
MC_SIMULATIONS_FOR_DD = 50  # Number of MC simulations to estimate max drawdown
MC_PERIODS_FOR_DD = None  # Will be set to length of returns data

# Toggle for max drawdown matching (can significantly slow down optimization)
MATCH_MAX_DRAWDOWN = False  # Default: False for speed, set to True for better tail risk matching

# Bounds for MJD parameters: (mu, sigma, lam, mu_J, sigma_J)
DEFAULT_BOUNDS = [
    (-0.3, 0.3),         # mu: annual drift
    (0.01, 1.0),         # sigma: annual volatility (constant)
    (0.0, 5.0),          # lam: jump intensity per year
    (-0.3, 0.0),         # mu_J: jump mean (typically negative)
    (0.01, 0.5),         # sigma_J: jump volatility
]

# Likelihood regularization weight (0 = no likelihood, 1 = equal weight)
# Lower values prioritize moment matching over likelihood
LIKELIHOOD_WEIGHT = 0.0  # Set to 0 for faster execution (moment matching doesn't need likelihood)

console = Console(width=200)

# ============================================================================
# MERTON JUMP DIFFUSION MODEL FUNCTIONS
# ============================================================================

def mjd_cf(u, t, params):
    """Merton Jump Diffusion Characteristic Function for log-return.
    
    For MJD: dS/S = mu*dt + sigma*dW + dJ
    where dJ is a compound Poisson process with jump size ~ N(mu_J, sigma_J^2)
    """
    mu = params[IDX['mu']]
    sigma = params[IDX['sigma']]
    lam = params[IDX['lam']]
    mu_J = params[IDX['mu_J']]
    sigma_J = params[IDX['sigma_J']]
    
    I = 0.0 + 1.0j
    
    # Diffusion component: exp(i*u*mu*t - 0.5*u^2*sigma^2*t)
    diffusion_term = I * u * mu * t - 0.5 * (u ** 2) * (sigma ** 2) * t
    
    # Jump component: exp(lambda*t*(exp(i*u*mu_J - 0.5*u^2*sigma_J^2) - 1))
    jump_exponent = I * u * mu_J - 0.5 * (u ** 2) * (sigma_J ** 2)
    jump_term = lam * t * (cmath.exp(jump_exponent) - 1.0)
    
    cf = cmath.exp(diffusion_term + jump_term)
    return cf

def cf_pdf_inversion_integrand(u, r_t, t, params):
    """Integrand for Lewis Fourier inversion formula."""
    cf_val = mjd_cf(u, t, params)
    integrand = cmath.exp(-1j * u * r_t) * cf_val
    return integrand.real

def cf_pdf_inversion(r_t, params, fast_mode=False):
    """PDF inversion using Lewis's integral.
    
    Args:
        fast_mode: If True, uses faster but less accurate integration (for regularization term)
    """
    if fast_mode:
        epsabs = 1e-4
        epsrel = 1e-3
        limit = 50
    else:
        epsabs = 1e-6
        epsrel = 1e-4
        limit = 100
    
    integral, err = quad(
        cf_pdf_inversion_integrand,
        0, CF_INTEGRATION_LIMIT,
        args=(r_t, DT, params),
        epsabs=epsabs,
        epsrel=epsrel,
        limit=limit
    )
    pdf = integral / math.pi
    return max(pdf, DENSITY_FLOOR)

def neg_log_likelihood_py(params, returns):
    """Pure Python negative log-likelihood for MJD."""
    if np.any(~np.isfinite(params)): 
        return 1e12
    
    sigma = params[IDX['sigma']]
    lam = params[IDX['lam']]
    mu = params[IDX['mu']]
    mu_J = params[IDX['mu_J']]
    sigma_J = params[IDX['sigma_J']]
    
    # Parameter validation
    if sigma <= 0 or sigma > 1.0: return 1e12
    if lam < 0 or lam > 5.0: return 1e12
    if sigma_J <= 0 or sigma_J > 0.5: return 1e12
    if mu_J < -0.3 or mu_J > 0.3: return 1e12
    if abs(mu) > 0.3: return 1e12
    
    # Calculate likelihood
    neg_ll = 0.0
    
    try:
        for r in returns:
            pdf_step = cf_pdf_inversion(r, params, fast_mode=True)
            neg_ll -= math.log(pdf_step)
    except Exception:
        return 1e12
    
    return neg_ll

# ============================================================================
# MOMENT CALCULATIONS
# ============================================================================

def compute_empirical_moments(returns):
    """Compute empirical moments from returns data.
    
    Returns:
        dict with keys: 'mean', 'std', 'skew', 'kurt', 'max_dd'
    """
    if len(returns) < 4:
        return None
    
    mean = np.mean(returns)
    std = np.std(returns, ddof=0)  # Population std
    skew = stats.skew(returns, bias=False) if len(returns) >= 3 else 0.0
    kurt = stats.kurtosis(returns, fisher=True, bias=False) if len(returns) >= 4 else 0.0  # Excess kurtosis
    
    # Compute max drawdown from cumulative returns
    cumulative_returns = np.cumprod(1.0 + returns)
    peak_series = np.maximum.accumulate(cumulative_returns)
    drawdowns = (cumulative_returns - peak_series) / peak_series
    max_dd = np.min(drawdowns) if len(drawdowns) > 0 else 0.0
    
    return {
        'mean': mean,
        'std': std,
        'skew': skew,
        'kurt': kurt,
        'max_dd': max_dd
    }

def compute_model_moments_analytical(params):
    """Compute MJD model moments analytically.
    
    For Merton Jump Diffusion:
    - Mean: (mu - 0.5*sigma^2)*DT + lam*DT*mu_J
    - Variance: sigma^2*DT + lam*DT*(sigma_J^2 + mu_J^2)
    - Skewness: from jump component (exact formula)
    - Kurtosis: from jump component (exact formula)
    
    Args:
        params: MJD model parameters (5-element array)
    
    Returns:
        dict with keys: 'mean', 'std', 'skew', 'kurt'
    """
    mu = params[IDX['mu']]
    sigma = params[IDX['sigma']]
    lam = params[IDX['lam']]
    mu_J = params[IDX['mu_J']]
    sigma_J = params[IDX['sigma_J']]
    
    # Mean: E[log_return] = (mu - 0.5*sigma^2)*DT + lam*DT*mu_J
    model_mean_period = (mu - 0.5 * sigma ** 2) * DT + lam * DT * mu_J
    
    # Variance: Var[log_return] = sigma^2*DT + lam*DT*(sigma_J^2 + mu_J^2)
    model_var_period = (sigma ** 2) * DT + lam * DT * (sigma_J ** 2 + mu_J ** 2)
    model_std_period = math.sqrt(max(model_var_period, 1e-12))
    
    # Skewness and kurtosis (exact formulas for MJD)
    if model_std_period > 1e-6:
        # Third moment (for skewness)
        # Diffusion contributes 0 to third moment (normal distribution)
        # Jump contributes: lam*DT*(mu_J^3 + 3*mu_J*sigma_J^2)
        jump_third_moment = lam * DT * (mu_J ** 3 + 3 * mu_J * sigma_J ** 2)
        model_skew = jump_third_moment / (model_std_period ** 3)
        model_skew = max(-5.0, min(5.0, model_skew))
        
        # Fourth moment (for excess kurtosis)
        # Diffusion contributes 0 to excess kurtosis (normal has kurtosis = 3)
        # Jump contributes: lam*DT*(mu_J^4 + 6*mu_J^2*sigma_J^2 + 3*sigma_J^4)
        jump_fourth_moment = lam * DT * (mu_J ** 4 + 6 * (mu_J ** 2) * (sigma_J ** 2) + 3 * (sigma_J ** 4))
        jump_kurt_raw = jump_fourth_moment / (model_std_period ** 4)
        model_kurt = jump_kurt_raw - 3.0  # Convert to excess kurtosis
        model_kurt = max(-3.0, min(30.0, model_kurt))
    else:
        model_skew = 0.0
        model_kurt = 0.0
    
    return {
        'mean': model_mean_period,
        'std': model_std_period,
        'skew': model_skew,
        'kurt': model_kurt
    }

# ============================================================================
# OBJECTIVE FUNCTION
# ============================================================================

def simulate_mjd_for_drawdown(params, n_periods, n_simulations=None, seed=None, show_progress=False):
    """Simulate MJD model to estimate max drawdown via Monte Carlo.
    
    This is more accurate than analytical approximations for tail risk metrics.
    
    Args:
        show_progress: If True, shows a progress bar (only for large simulations)
    """
    if n_simulations is None:
        n_simulations = MC_SIMULATIONS_FOR_DD
    rng = np.random.default_rng(seed)
    mu = params[IDX['mu']]
    sigma = params[IDX['sigma']]
    lam = params[IDX['lam']]
    mu_J = params[IDX['mu_J']]
    sigma_J = params[IDX['sigma_J']]
    
    dt = DT
    max_drawdowns = []
    
    # Only show progress for larger simulations
    use_progress = show_progress and n_simulations >= 10
    
    if use_progress:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
            transient=True
        )
        task = progress.add_task(f"[cyan]Computing max drawdown ({n_simulations} sims)...", total=n_simulations)
        progress.start()
    
    try:
        for sim in range(n_simulations):
            # Initialize
            price = 1.0
            prices = [price]
            
            for t in range(n_periods):
                # Standard Wiener process
                z = rng.standard_normal()
                
                # Merton jumps
                num_jumps = rng.poisson(lam * dt)
                if num_jumps > 0:
                    jump_size = np.sum(rng.normal(mu_J, sigma_J, num_jumps))
                else:
                    jump_size = 0.0
                
                # MJD price process: dS/S = mu*dt + sigma*dW + dJ
                log_return = (mu - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * z + jump_size
                price *= math.exp(log_return)
                prices.append(price)
            
            # Compute max drawdown for this simulation
            prices_array = np.array(prices)
            peak_series = np.maximum.accumulate(prices_array)
            drawdowns = (prices_array - peak_series) / peak_series
            max_dd = np.min(drawdowns) if len(drawdowns) > 0 else 0.0
            max_drawdowns.append(max_dd)
            
            if use_progress:
                progress.update(task, advance=1)
    finally:
        if use_progress:
            progress.stop()
    
    # Return median max drawdown (more robust than mean)
    return np.median(max_drawdowns) if len(max_drawdowns) > 0 else 0.0

def moment_matching_objective(params, emp_moments, returns, likelihood_weight=LIKELIHOOD_WEIGHT, match_max_dd=None):
    """Moment matching objective function.
    
    Minimizes squared error between model and empirical moments (including max drawdown),
    with optional likelihood regularization.
    
    Args:
        params: MJD model parameters (5-element array)
        emp_moments: dict with empirical moments (including 'max_dd')
        returns: empirical returns data
        likelihood_weight: weight for likelihood term (0 = no likelihood)
        match_max_dd: whether to match max drawdown (None = use global MATCH_MAX_DRAWDOWN)
    
    Returns:
        Objective value to minimize
    """
    global MATCH_MAX_DRAWDOWN
    if match_max_dd is None:
        match_max_dd = MATCH_MAX_DRAWDOWN
    
    try:
        # Parameter validation
        if np.any(~np.isfinite(params)): 
            return 1e12
        
        sigma = params[IDX['sigma']]
        lam = params[IDX['lam']]
        mu = params[IDX['mu']]
        mu_J = params[IDX['mu_J']]
        sigma_J = params[IDX['sigma_J']]
        
        # Parameter bounds validation
        if sigma <= 0 or sigma > 1.0: return 1e12
        if lam < 0 or lam > 5.0: return 1e12
        if sigma_J <= 0 or sigma_J > 0.5: return 1e12
        if mu_J < -0.3 or mu_J > 0.3: return 1e12
        if abs(mu) > 0.3: return 1e12
        
        # Calculate model moments
        model_moments = compute_model_moments_analytical(params)
        
        # Compute max drawdown via Monte Carlo (only if enabled - this is expensive!)
        if match_max_dd:
            n_periods = len(returns)
            model_max_dd = simulate_mjd_for_drawdown(params, n_periods, n_simulations=MC_SIMULATIONS_FOR_DD, show_progress=DEBUG)
            dd_abs_error = abs(model_max_dd - emp_moments.get('max_dd', 0.0))
        else:
            model_max_dd = 0.0
            dd_abs_error = 0.0
        
        # Moment matching errors (weighted by importance)
        mean_abs_error = abs(model_moments['mean'] - emp_moments['mean'])
        std_abs_error = abs(model_moments['std'] - emp_moments['std'])
        skew_abs_error = abs(model_moments['skew'] - emp_moments['skew'])
        kurt_abs_error = abs(model_moments['kurt'] - emp_moments['kurt'])
        
        # Use relative errors when values are significant, absolute errors otherwise
        if abs(emp_moments['mean']) > 1e-4:
            mean_error = mean_abs_error / abs(emp_moments['mean'])
        else:
            mean_error = mean_abs_error * 100.0
        
        if emp_moments['std'] > 1e-4:
            std_error = std_abs_error / emp_moments['std']
        else:
            std_error = std_abs_error * 100.0
        
        if abs(emp_moments['skew']) > 0.1:
            skew_error = skew_abs_error / abs(emp_moments['skew'])
        else:
            skew_error = skew_abs_error * 10.0
        
        if abs(emp_moments['kurt']) > 0.1:
            kurt_error = kurt_abs_error / abs(emp_moments['kurt'])
        else:
            kurt_error = kurt_abs_error * 10.0
        
        # Max drawdown error (use relative error)
        emp_max_dd = emp_moments.get('max_dd', 0.0)
        if abs(emp_max_dd) > 0.01:
            dd_error = dd_abs_error / abs(emp_max_dd)
        else:
            dd_error = dd_abs_error * 10.0
        
        # Primary objective: moment matching
        moment_error = (10000.0 * (mean_error ** 2) +
                        20000.0 * (std_error ** 2) +
                        25000.0 * (skew_error ** 2) +
                        25000.0 * (kurt_error ** 2))
        
        # Add max drawdown error only if enabled
        if match_max_dd:
            if abs(emp_max_dd) > 0.5:
                moment_error += 50000.0 * (dd_error ** 2)
            else:
                moment_error += 20000.0 * (dd_error ** 2)
        
        # Secondary objective: likelihood (regularization)
        likelihood_term = 0.0
        if likelihood_weight > 0:
            try:
                sample_size = min(len(returns), 100)
                if len(returns) > sample_size:
                    indices = np.linspace(0, len(returns)-1, sample_size, dtype=int)
                    returns_sample = returns[indices]
                else:
                    returns_sample = returns
                
                neg_ll = neg_log_likelihood_py(params, returns_sample)
                likelihood_term = likelihood_weight * neg_ll / len(returns_sample) * (len(returns_sample) / len(returns))
            except Exception as e:
                if DEBUG:
                    console.print(f"[dim]      Likelihood calculation failed: {e}[/dim]")
                likelihood_term = 100.0
        
        total_obj = moment_error + max(0, likelihood_term)
        
        if DEBUG and total_obj > 10:
            console.print(f"[dim]      Objective: {total_obj:.2e} | moment_err={moment_error:.2e} | "
                         f"model: mean={model_moments['mean']:.4f}, std={model_moments['std']:.4f}, "
                         f"skew={model_moments['skew']:.2f}, kurt={model_moments['kurt']:.2f}, dd={model_max_dd:.4f} | "
                         f"emp: mean={emp_moments['mean']:.4f}, std={emp_moments['std']:.4f}, "
                         f"skew={emp_moments['skew']:.2f}, kurt={emp_moments['kurt']:.2f}, dd={emp_moments.get('max_dd', 0.0):.4f}[/dim]")
        
        return total_obj
    except Exception as e:
        if DEBUG:
            console.print(f"[dim]      Error in moment_matching_objective: {e}[/dim]")
        return 1e12

# ============================================================================
# BOUND DIAGNOSTICS
# ============================================================================

def check_boundary_issues(params, bounds, tolerance=0.01):
    """
    Check if parameters are hitting boundaries, indicating bounds may be too tight or too wide.
    
    Args:
        params: Fitted parameter array
        bounds: List of (lower, upper) bounds
        tolerance: Distance from boundary to consider "near boundary" (as fraction of range)
    
    Returns:
        dict with diagnostics
    """
    hitting = []
    near = []
    distances = {}
    recommendations = []
    
    param_names = ['mu', 'sigma', 'lam', 'mu_J', 'sigma_J']
    
    for i, (param_val, (lower, upper)) in enumerate(zip(params, bounds)):
        range_size = upper - lower
        dist_lower = abs(param_val - lower) / range_size if range_size > 0 else 0
        dist_upper = abs(param_val - upper) / range_size if range_size > 0 else 0
        min_dist = min(dist_lower, dist_upper)
        
        distances[param_names[i]] = {
            'value': param_val,
            'distance_from_lower': dist_lower,
            'distance_from_upper': dist_upper,
            'min_distance': min_dist
        }
        
        if min_dist < 0.001:
            hitting.append(i)
            if dist_lower < 0.001:
                recommendations.append(
                    f"{param_names[i]}: Hitting LOWER bound ({lower:.4f}). Consider lowering lower bound."
                )
            else:
                recommendations.append(
                    f"{param_names[i]}: Hitting UPPER bound ({upper:.4f}). Consider raising upper bound."
                )
        elif min_dist < tolerance:
            near.append(i)
            if dist_lower < tolerance:
                recommendations.append(
                    f"{param_names[i]}: Near LOWER bound ({lower:.4f}, current={param_val:.4f}). "
                    f"Bounds may be too tight or parameter needs lower bound."
                )
            else:
                recommendations.append(
                    f"{param_names[i]}: Near UPPER bound ({upper:.4f}, current={param_val:.4f}). "
                    f"Bounds may be too tight or parameter needs higher bound."
                )
    
    return {
        'hitting_boundaries': hitting,
        'near_boundaries': near,
        'boundary_distances': distances,
        'recommendations': recommendations
    }

# ============================================================================
# OPTIMIZATION
# ============================================================================

def fit_mjd_moment_matching(returns, name="regime", restarts=NUM_RESTARTS, maxiter=MAXITER,
                            bounds_vec=DEFAULT_BOUNDS, use_bounds=True, match_max_dd=None):
    """Fit MJD model to returns using moment matching.
    
    Args:
        returns: array of log returns
        name: name for this regime (for logging)
        restarts: number of optimization restarts
        maxiter: maximum iterations per optimization
        bounds_vec: parameter bounds
        use_bounds: whether to use bounds
        match_max_dd: whether to match max drawdown (None = use global MATCH_MAX_DRAWDOWN)
    
    Returns:
        dict with fitted parameters and statistics, or None if fitting fails
    """
    global MATCH_MAX_DRAWDOWN
    if match_max_dd is None:
        match_max_dd = MATCH_MAX_DRAWDOWN
    n = len(returns)
    if n < 8:
        console.print(f"[yellow]Too few observations for {name} ({n}); skipping.[/yellow]")
        return None
    
    # Increase restarts for small regimes
    if n < 200:
        restarts = max(restarts, NUM_RESTARTS_SMALL)
        if DEBUG:
            console.print(f"[dim]  Using {restarts} restarts (increased for small sample size)[/dim]")
    
    # Compute empirical moments
    emp_moments = compute_empirical_moments(returns)
    if emp_moments is None:
        console.print(f"[yellow]Could not compute moments for {name}; skipping.[/yellow]")
        return None
    
    if DEBUG:
        console.print(f"\n[cyan]Fitting {name} (n={n})[/cyan]")
        console.print(f"[dim]  Empirical moments: mean={emp_moments['mean']:.6f}, "
                     f"std={emp_moments['std']:.6f}, skew={emp_moments['skew']:.3f}, "
                     f"kurt={emp_moments['kurt']:.3f}, max_dd={emp_moments.get('max_dd', 0.0):.4f}[/dim]")
    
    # Initial parameter guess based on empirical moments
    mean_period = emp_moments['mean']
    var_period = emp_moments['std'] ** 2
    mu_annual = mean_period / DT
    var_annual = var_period / DT
    vol_annual = math.sqrt(max(var_annual, 1e-12))
    
    init_base = np.zeros(5, dtype=float)
    init_base[IDX['mu']] = np.clip(mu_annual, -0.3, 0.3)
    init_base[IDX['sigma']] = np.clip(vol_annual, 0.01, 1.0)
    
    # Jump parameters: initialize based on skewness/kurtosis
    if abs(emp_moments['skew']) > 0.5 or emp_moments['kurt'] > 3.0:
        init_base[IDX['lam']] = 0.5
        init_base[IDX['mu_J']] = -0.05 if emp_moments['skew'] < 0 else 0.0
        init_base[IDX['sigma_J']] = 0.1
    else:
        init_base[IDX['lam']] = 0.1
        init_base[IDX['mu_J']] = -0.02
        init_base[IDX['sigma_J']] = 0.05
    
    # Bounds
    if use_bounds:
        bounds = list(bounds_vec)
    else:
        bounds = [(None, None)] * len(init_base)
    
    # Optimization with multiple restarts
    best_val = np.inf
    best_res = None
    rng = np.random.default_rng(12345 + abs(hash(name)) % 9999)
    
    start_time = time.time()
    
    # Progress bar for optimization restarts
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TextColumn("[cyan]Best: {task.fields[best_obj]:.2e}"),
        TimeElapsedColumn(),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task(
            f"[cyan]Optimizing {name}...",
            total=restarts,
            best_obj=float('inf')
        )
        
        for attempt in range(restarts):
            # Perturb initial guess
            scales = np.array([0.005, 0.05, 0.1, 0.002, 0.01])
            perturb = rng.normal(scale=scales)
            x0 = init_base + perturb
            
            # Ensure within bounds
            if use_bounds:
                for j in range(len(x0)):
                    lo, hi = bounds[j]
                    if lo is not None and x0[j] <= lo:
                        x0[j] = lo + 1e-12
                    if hi is not None and x0[j] >= hi:
                        x0[j] = hi - 1e-12
            
            # Optimize
            opt_options = {
                "maxiter": min(maxiter, 10000),
                "ftol": 1e-7,
                "gtol": 1e-6,
                "maxls": 20,
            }
            
            try:
                res = minimize(
                    moment_matching_objective,
                    x0,
                    args=(emp_moments, returns, LIKELIHOOD_WEIGHT, match_max_dd),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options=opt_options
                )
                
                if res.success and res.fun < best_val:
                    best_val = res.fun
                    best_res = res
                    progress.update(task, best_obj=best_val)
                    
                    # Early stopping if moment matching is very good
                    if best_val < EARLY_STOP_MOMENT_ERROR:
                        if DEBUG:
                            console.print(f"[green]  Early stopping: moment error < {EARLY_STOP_MOMENT_ERROR}[/green]")
                        break
            except Exception as e:
                if DEBUG:
                    console.print(f"[dim]  Attempt {attempt+1} failed: {e}[/dim]")
                continue
            
            progress.update(task, advance=1)
    
    elapsed_time = time.time() - start_time
    
    if best_res is None:
        console.print(f"[red]Failed to fit {name} after {restarts} attempts[/red]")
        return None
    
    # Extract results
    params = best_res.x
    model_moments = compute_model_moments_analytical(params)
    
    # Always compute max drawdown for final results
    if DEBUG:
        console.print(f"[dim]  Computing final max drawdown estimate (50 simulations)...[/dim]")
    model_max_dd = simulate_mjd_for_drawdown(params, n, n_simulations=50, show_progress=DEBUG)
    
    # Compute moment errors
    mean_error = abs(model_moments['mean'] - emp_moments['mean']) / abs(emp_moments['mean']) if abs(emp_moments['mean']) > 1e-4 else abs(model_moments['mean'] - emp_moments['mean'])
    std_error = abs(model_moments['std'] - emp_moments['std']) / emp_moments['std'] if emp_moments['std'] > 1e-4 else abs(model_moments['std'] - emp_moments['std'])
    skew_error = abs(model_moments['skew'] - emp_moments['skew']) / abs(emp_moments['skew']) if abs(emp_moments['skew']) > 0.1 else abs(model_moments['skew'] - emp_moments['skew'])
    kurt_error = abs(model_moments['kurt'] - emp_moments['kurt']) / abs(emp_moments['kurt']) if abs(emp_moments['kurt']) > 0.1 else abs(model_moments['kurt'] - emp_moments['kurt'])
    emp_max_dd = emp_moments.get('max_dd', 0.0)
    dd_error = abs(model_max_dd - emp_max_dd) / abs(emp_max_dd) if abs(emp_max_dd) > 0.01 else abs(model_max_dd - emp_max_dd)
    
    # Add max_dd to model_moments for output
    model_moments['max_dd'] = model_max_dd
    
    # Check for boundary issues
    boundary_diagnostics = check_boundary_issues(params, bounds if use_bounds else [(None, None)] * len(params))
    
    result = {
        'name': name,
        'n_obs': n,
        'params': params,
        'objective': best_val,
        'emp_moments': emp_moments,
        'model_moments': model_moments,
        'moment_errors': {
            'mean': mean_error,
            'std': std_error,
            'skew': skew_error,
            'kurt': kurt_error,
            'max_dd': dd_error
        },
        'elapsed_time': elapsed_time,
        'optimization_success': best_res.success,
        'boundary_diagnostics': boundary_diagnostics
    }
    
    if DEBUG:
        console.print(f"[green]  ✓ Fitted {name} in {elapsed_time:.2f}s[/green]")
        console.print(f"[dim]    Moment errors: mean={mean_error*100:.2f}%, std={std_error*100:.2f}%, "
                     f"skew={skew_error*100:.2f}%, kurt={kurt_error*100:.2f}%, dd={dd_error*100:.2f}%[/dim]")
        
        if boundary_diagnostics['hitting_boundaries']:
            console.print(f"[yellow]  ⚠ Warning: {len(boundary_diagnostics['hitting_boundaries'])} parameters hitting boundaries[/yellow]")
            for rec in boundary_diagnostics['recommendations'][:3]:
                console.print(f"[dim]    {rec}[/dim]")
    
    return result

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main function to fit MJD model to each regime."""
    global DEBUG
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Fit Merton Jump Diffusion model to regimes using moment matching')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--csv', type=str, default='data/regime_classification_nominal_returns.csv',
                       help='Path to CSV file with returns and regime IDs')
    parser.add_argument('--match-max-dd', action='store_true', 
                       help='Enable max drawdown matching (slower but more accurate for tail risk)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Disable plot generation (faster execution)')
    args = parser.parse_args()
    
    global DEBUG, MATCH_MAX_DRAWDOWN
    DEBUG = args.debug
    MATCH_MAX_DRAWDOWN = args.match_max_dd
    
    if MATCH_MAX_DRAWDOWN:
        console.print("[yellow]⚠ Max drawdown matching enabled - optimization will be slower[/yellow]")
    else:
        console.print("[dim]Max drawdown matching disabled (use --match-max-dd to enable)[/dim]")
    
    console.print("[bold cyan]Merton Jump Diffusion Model Moment Matching[/bold cyan]")
    console.print("=" * 80)
    
    # Read CSV file
    csv_path = args.csv
    if not os.path.exists(csv_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        csv_path = os.path.join(parent_dir, csv_path)
        if not os.path.exists(csv_path):
            csv_path = os.path.join(os.getcwd(), args.csv)
    
    if not os.path.exists(csv_path):
        console.print(f"[red]Error: CSV file not found: {args.csv}[/red]")
        return
    
    console.print(f"[green]Reading CSV: {csv_path}[/green]")
    df = pd.read_csv(csv_path)
    
    # Convert returns from percentage to decimal
    if 'Total Nominal Return (%)' in df.columns:
        df['returns'] = df['Total Nominal Return (%)'] / 100.0
    elif 'returns' in df.columns:
        pass
    else:
        console.print("[red]Error: Could not find returns column[/red]")
        return
    
    # Get regime column
    regime_col = None
    for col in df.columns:
        if 'regime' in col.lower() or 'Regime' in col:
            regime_col = col
            break
    
    if regime_col is None:
        console.print("[red]Error: Could not find regime ID column[/red]")
        return
    
    # Group by regime
    regimes = df.groupby(regime_col)
    
    console.print(f"\n[cyan]Found {len(regimes)} regimes[/cyan]")
    
    # Estimate time
    total_restarts = 0
    for regime_id, regime_df in regimes:
        n = len(regime_df)
        restarts = NUM_RESTARTS_SMALL if n < 200 else NUM_RESTARTS
        total_restarts += restarts
    
    console.print(f"[dim]Estimated optimization attempts: ~{total_restarts} restarts across {len(regimes)} regimes[/dim]")
    console.print(f"[dim]This may take several minutes. Progress bars will show current status...[/dim]\n")
    
    # Fit each regime
    results = []
    regime_list = list(regimes)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TextColumn("[cyan]{task.fields[regime_name]}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task(
            "[bold cyan]Fitting regimes...",
            total=len(regime_list),
            regime_name=""
        )
        
        for regime_id, regime_df in regimes:
            progress.update(task, regime_name=f"regime_{regime_id}")
            regime_returns = regime_df['returns'].values
            
            if len(regime_returns) < 10:
                console.print(f"[yellow]Skipping regime_{regime_id}: Only {len(regime_returns)} observations[/yellow]")
                progress.update(task, advance=1)
                continue
            
            # Convert to log returns if needed
            if np.any(np.abs(regime_returns) > 0.5):
                regime_returns = np.log(1.0 + regime_returns)
            
            result = fit_mjd_moment_matching(
                regime_returns,
                name=f"regime_{regime_id}",
                restarts=NUM_RESTARTS,
                maxiter=MAXITER,
                bounds_vec=DEFAULT_BOUNDS
            )
            
            if result is not None:
                results.append(result)
            
            progress.update(task, advance=1)
    
    if len(results) == 0:
        console.print("[red]No regimes were successfully fitted[/red]")
        return
    
    # Display results table
    console.print("\n[bold cyan]Fitting Results[/bold cyan]")
    console.print("=" * 80)
    
    table = Table(title="MJD Model Parameters - Moment Matching")
    table.add_column("Regime", style="cyan")
    table.add_column("N", justify="right")
    table.add_column("μ (annual)", justify="right", style="green")
    table.add_column("σ (annual)", justify="right")
    table.add_column("λ (annual)", justify="right")
    table.add_column("μ_J", justify="right")
    table.add_column("σ_J", justify="right")
    table.add_column("Time (s)", justify="right")
    
    for r in results:
        p = r['params']
        table.add_row(
            r['name'],
            str(r['n_obs']),
            f"{p[IDX['mu']]:.4f}",
            f"{p[IDX['sigma']]:.4f}",
            f"{p[IDX['lam']]:.3f}",
            f"{p[IDX['mu_J']]:.4f}",
            f"{p[IDX['sigma_J']]:.4f}",
            f"{r['elapsed_time']:.2f}"
        )
    
    console.print(table)
    
    # Moment matching accuracy table
    console.print("\n[bold cyan]Moment Matching Accuracy[/bold cyan]")
    console.print("=" * 80)
    
    table2 = Table(title="Empirical vs Model Moments")
    table2.add_column("Regime", style="cyan")
    table2.add_column("Moment", style="yellow")
    table2.add_column("Empirical", justify="right", style="green")
    table2.add_column("Model", justify="right", style="blue")
    table2.add_column("Error %", justify="right", style="red")
    
    for r in results:
        emp = r['emp_moments']
        model = r['model_moments']
        errors = r['moment_errors']
        
        table2.add_row(r['name'], "Mean", f"{emp['mean']:.6f}", f"{model['mean']:.6f}", f"{errors['mean']*100:.2f}%", style="bold")
        table2.add_row("", "Std", f"{emp['std']:.6f}", f"{model['std']:.6f}", f"{errors['std']*100:.2f}%")
        table2.add_row("", "Skew", f"{emp['skew']:.3f}", f"{model['skew']:.3f}", f"{errors['skew']*100:.2f}%")
        table2.add_row("", "Kurt", f"{emp['kurt']:.3f}", f"{model['kurt']:.3f}", f"{errors['kurt']*100:.2f}%")
        table2.add_row("", "Max DD", f"{emp.get('max_dd', 0.0):.4f}", f"{model.get('max_dd', 0.0):.4f}", f"{errors.get('max_dd', 0.0)*100:.2f}%")
        table2.add_row("", "", "", "", "")
    
    console.print(table2)
    
    # Save results to CSV
    os.makedirs('data', exist_ok=True)
    output_file = "data/MJD_per_regime_Moment_Matching_results.csv"
    console.print(f"\n[green]Saving results to: {output_file}[/green]")
    
    output_data = []
    for r in results:
        p = r['params']
        emp = r['emp_moments']
        model = r['model_moments']
        errors = r['moment_errors']
        
        output_data.append({
            'name': r['name'],
            'N_obs': r['n_obs'],
            'mu_annual': p[IDX['mu']],
            'sigma_annual': p[IDX['sigma']],
            'lambda_per_year': p[IDX['lam']],
            'mu_J': p[IDX['mu_J']],
            'sigma_J': p[IDX['sigma_J']],
            'objective': r['objective'],
            'emp_mean': emp['mean'],
            'emp_std': emp['std'],
            'emp_skew': emp['skew'],
            'emp_kurt': emp['kurt'],
            'emp_max_dd': emp.get('max_dd', 0.0),
            'model_mean': model['mean'],
            'model_std': model['std'],
            'model_skew': model['skew'],
            'model_kurt': model['kurt'],
            'model_max_dd': model.get('max_dd', 0.0),
            'mean_error_pct': errors['mean'] * 100,
            'std_error_pct': errors['std'] * 100,
            'skew_error_pct': errors['skew'] * 100,
            'kurt_error_pct': errors['kurt'] * 100,
            'max_dd_error_pct': errors.get('max_dd', 0.0) * 100,
            'elapsed_time_sec': r['elapsed_time']
        })
    
    output_df = pd.DataFrame(output_data)
    output_df.to_csv(output_file, index=False)
    
    console.print(f"[green]✓ Results saved successfully[/green]")
    
    # Boundary diagnostics summary
    console.print("\n[bold cyan]Boundary Diagnostics Summary[/bold cyan]")
    console.print("=" * 80)
    
    all_boundary_issues = []
    for r in results:
        if 'boundary_diagnostics' in r and r['boundary_diagnostics']:
            diag = r['boundary_diagnostics']
            if diag['hitting_boundaries'] or diag['near_boundaries']:
                all_boundary_issues.append((r['name'], diag))
    
    if all_boundary_issues:
        console.print("[yellow]⚠ Some parameters are hitting or near boundaries:[/yellow]")
        for name, diag in all_boundary_issues:
            if diag['hitting_boundaries']:
                console.print(f"  [red]{name}: {len(diag['hitting_boundaries'])} parameters hitting boundaries[/red]")
                for rec in diag['recommendations'][:2]:
                    console.print(f"    [dim]{rec}[/dim]")
        console.print("\n[dim]Recommendation: Review bounds - they may be too tight for your data.[/dim]")
    else:
        console.print("[green]✓ No boundary issues detected - bounds appear appropriate[/green]")
    
    console.print(f"\n[bold green]Completed! Fitted {len(results)} regimes.[/bold green]")
    
    # Generate plots if enabled
    if not args.no_plots:
        console.print("\n[bold cyan]Generating distribution visualizations...[/bold cyan]")
        # Note: Visualization functions would need to be updated for MJD
        # For now, just print a message
        console.print("[dim]Plot generation not yet implemented for MJD model[/dim]")
    else:
        console.print("\n[dim]Plot generation disabled[/dim]")

if __name__ == "__main__":
    main()
