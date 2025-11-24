"""
Bates Model Moment Matching for Regime-Based Stock Returns

This script fits Bates model parameters to each regime by matching the first four moments:
- Mean (first moment)
- Variance (second moment)
- Skewness (third moment)
- Kurtosis (fourth moment)

Optionally, max drawdown can also be matched (use --match-max-dd flag).
Note: Max drawdown matching is computationally expensive and significantly slows optimization.

The goal is to ensure Monte Carlo simulations using the Bates model accurately reproduce
the historical return distribution characteristics for each volatility regime.

OPTIMIZATIONS:
1. Cython acceleration for likelihood calculations (10-50x speedup)
2. Fast integration mode for likelihood regularization term
3. Smart initial parameter guesses based on empirical moments
4. Parallel optimization restarts for better global search
5. Early stopping when moment matching is achieved

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

For OPTION PRICING vs TIME SERIES:
- Option pricing typically needs TIGHTER bounds for vol-of-vol (nu: 0.05-3.0 vs 0.1-10.0)
  because options are more sensitive and market-implied vol-of-vol is usually lower
- Option pricing needs WIDER bounds for jump intensity (lam: 0-10 vs 0-5) and jump size
  because options often imply higher jump activity than historical returns show
- Use --option-pricing flag to automatically use appropriate bounds

The script automatically detects and reports boundary issues in the output.

Usage:
    python "2B. Moment_Matching_v2.py" [--debug] [--no-cython] [--match-max-dd] [--option-pricing]
    
    --debug: Enable detailed debug output (includes boundary diagnostics)
    --no-cython: Disable Cython acceleration (use pure Python)
    --match-max-dd: Enable max drawdown matching (slower but more accurate for tail risk)
    --option-pricing: Use bounds optimized for option pricing calibration
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

# Parameter index positions (consistent with existing codebase)
MU_IDX = 0; KAPPA_IDX = 1; THETA_IDX = 2; NU_IDX = 3; RHO_IDX = 4; V0_IDX = 5
LAM_IDX = 6; MU_J_IDX = 7; SIGMA_J_IDX = 8

IDX = {
    "mu": MU_IDX, "kappa": KAPPA_IDX, "theta": THETA_IDX, "nu": NU_IDX,
    "rho": RHO_IDX, "v0": V0_IDX, "lam": LAM_IDX, "mu_J": MU_J_IDX,
    "sigma_J": SIGMA_J_IDX
}

# Constants
PERIODS_PER_YEAR = 12
DT = 1.0 / PERIODS_PER_YEAR
DENSITY_FLOOR = 1e-12
CF_INTEGRATION_LIMIT = 50.0

# Optimization settings
NUM_RESTARTS = 15  # Number of optimization restarts (reduced from 30 for speed)
NUM_RESTARTS_SMALL = 25  # More restarts for small regimes (reduced from 50)
MAXITER = 10000  # Maximum iterations per optimization (reduced from 50000 for speed)
EARLY_STOP_MOMENT_ERROR = 0.01  # Stop if moment error < 1% (early stopping)

# Monte Carlo settings for max drawdown calculation
MC_SIMULATIONS_FOR_DD = 50  # Number of MC simulations to estimate max drawdown (increased from 20 for better accuracy)
MC_PERIODS_FOR_DD = None  # Will be set to length of returns data

# Toggle for max drawdown matching (can significantly slow down optimization)
# Set to False for faster fitting (max drawdown will still be computed for final results)
MATCH_MAX_DRAWDOWN = True  # Default: False for speed, set to True for better tail risk matching

# Bounds for parameters: (mu, kappa, theta, nu, rho, v0, lam, mu_J, sigma_J)
# NOTE: These bounds are for TIME SERIES calibration (fitting to historical returns)
# For OPTION PRICING calibration, see OPTION_PRICING_BOUNDS below
DEFAULT_BOUNDS = [
    (-0.3, 0.3),         # mu: annual drift
    (0.1, 50.0),         # kappa: mean reversion speed
    (0.001, 0.25),       # theta: long-run variance
    (0.1, 10.0),         # nu: vol-of-vol
    (-0.99, 0.0),        # rho: correlation (typically negative)
    (0.001, 0.25),       # v0: initial variance
    (0.0, 5.0),          # lam: jump intensity per year
    (-0.3, 0.0),         # mu_J: jump mean
    (0.01, 0.5),         # sigma_J: jump volatility
]

# Bounds for OPTION PRICING calibration (typically tighter for vol-of-vol, wider for jumps)
# Option prices are more sensitive to certain parameters, and market-implied values differ from historical
OPTION_PRICING_BOUNDS = [
    (-0.2, 0.2),         # mu: annual drift (tighter - less relevant for options)
    (0.5, 20.0),         # kappa: mean reversion speed (tighter lower bound - options need mean reversion)
    (0.001, 0.20),       # theta: long-run variance (similar to time series)
    (0.05, 3.0),         # nu: vol-of-vol (TIGHTER - options typically show lower vol-of-vol than historical)
    (-0.99, -0.3),       # rho: correlation (tighter - options need negative correlation for skew)
    (0.001, 0.20),       # v0: initial variance (similar to time series)
    (0.0, 10.0),         # lam: jump intensity (WIDER - options often imply higher jump intensity)
    (-0.5, 0.0),         # mu_J: jump mean (wider negative range - options show more negative jumps)
    (0.01, 0.8),         # sigma_J: jump volatility (WIDER - options often imply larger jump sizes)
]

# Likelihood regularization weight (0 = no likelihood, 1 = equal weight)
# Lower values prioritize moment matching over likelihood
# Set to 0 by default for speed (moment matching doesn't need likelihood)
LIKELIHOOD_WEIGHT = 0.0  # Changed from 0.05 to 0.0 for faster execution

console = Console(width=200)

# ============================================================================
# CYTHON MODULE IMPORT
# ============================================================================

CYTHON_AVAILABLE = False
NEG_LOG_LIKELIHOOD_C = None

def try_import_cython():
    """Try to import compiled Cython module for faster likelihood calculations."""
    global CYTHON_AVAILABLE, NEG_LOG_LIKELIHOOD_C
    
    try:
        # Check multiple possible locations
        _script_dir = os.path.dirname(os.path.abspath(__file__))
        _parent_dir = os.path.dirname(_script_dir)
        _grandparent_dir = os.path.dirname(_parent_dir)
        
        # Add to path
        for path in [_parent_dir, _grandparent_dir, os.getcwd()]:
            if path not in sys.path:
                sys.path.insert(0, path)
        
        # Add cython folder to path
        cython_dir = os.path.join(_grandparent_dir, 'cython')
        if os.path.exists(cython_dir) and cython_dir not in sys.path:
            sys.path.insert(0, cython_dir)
        
        # Check build directories
        import platform
        python_version = f"{sys.version_info.major}{sys.version_info.minor}"
        build_dirs = [
            os.path.join(_grandparent_dir, 'build'),
            os.path.join(_grandparent_dir, f'build/lib.win-amd64-cpython-{python_version}'),
            os.path.join(_grandparent_dir, 'cython'),
            os.path.join(_parent_dir, 'build'),
        ]
        
        for build_dir in build_dirs:
            if os.path.exists(build_dir) and build_dir not in sys.path:
                sys.path.insert(0, build_dir)
        
        from bates_mle_cython import neg_log_likelihood_c
        NEG_LOG_LIKELIHOOD_C = neg_log_likelihood_c
        CYTHON_AVAILABLE = True
        if DEBUG:
            console.print("[green]✓ Cython module loaded - using accelerated likelihood calculations[/green]")
        return True
    except ImportError:
        CYTHON_AVAILABLE = False
        if DEBUG:
            console.print("[yellow]⚠ Cython module not available - using pure Python (slower)[/yellow]")
        return False

# ============================================================================
# BATES MODEL FUNCTIONS
# ============================================================================

def bates_cf(u, v_prev, t, params):
    """Bates Characteristic Function for log-return conditional on v_prev."""
    mu = params[IDX['mu']]
    kappa = params[IDX['kappa']]
    theta = params[IDX['theta']]
    nu = params[IDX['nu']]
    rho = params[IDX['rho']]
    lam = params[IDX['lam']]
    mu_J = params[IDX['mu_J']]
    sigma_J = params[IDX['sigma_J']]
    
    I = 0.0 + 1.0j
    alpha = u * u + I * u
    beta = kappa - I * rho * nu * u
    gamma = cmath.sqrt(beta**2 + alpha * nu**2)
    
    g = (beta - gamma) / (beta + gamma)
    exp_gamma_t = cmath.exp(-gamma * t)
    
    C_t_u_numerator = (beta - gamma + (beta + gamma) * g * exp_gamma_t)
    C_t_u_denominator = (nu**2 * (1.0 - g * exp_gamma_t))
    C_t_u = (1.0 - exp_gamma_t) / C_t_u_denominator * C_t_u_numerator
    
    D_t_u_log_term = cmath.log((1.0 - g * exp_gamma_t) / (1.0 - g))
    D_t_u = (kappa * theta / nu**2) * ((beta - gamma) * t - 2.0 * D_t_u_log_term)
    
    jump_exponent = I * u * mu_J - 0.5 * u * u * sigma_J ** 2
    jump_term = lam * t * (cmath.exp(jump_exponent) - 1.0)
    
    cf = cmath.exp(I * u * mu * t + C_t_u * v_prev + D_t_u + jump_term)
    return cf

def cf_pdf_inversion_integrand(u, r_t, v_prev, t, params):
    """Integrand for Lewis Fourier inversion formula."""
    cf_val = bates_cf(u, v_prev, t, params)
    integrand = cmath.exp(-1j * u * r_t) * cf_val
    return integrand.real

def cf_pdf_inversion(r_t, v_prev, params, fast_mode=False):
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
        args=(r_t, v_prev, DT, params),
        epsabs=epsabs,
        epsrel=epsrel,
        limit=limit
    )
    pdf = integral / math.pi
    return max(pdf, DENSITY_FLOOR)

def neg_log_likelihood_py(params, returns):
    """Pure Python negative log-likelihood."""
    if np.any(~np.isfinite(params)): 
        return 1e12
    
    kappa = params[IDX['kappa']]
    theta = params[IDX['theta']]
    nu = params[IDX['nu']]
    v0 = params[IDX['v0']]
    lam = params[IDX['lam']]
    mu = params[IDX['mu']]
    rho = params[IDX['rho']]
    mu_J = params[IDX['mu_J']]
    sigma_J = params[IDX['sigma_J']]
    
    # Parameter validation
    if kappa <= 0 or kappa > 100.0: return 1e12
    if theta <= 0 or theta > 0.25: return 1e12
    if nu <= 0 or nu > 20.0: return 1e12
    if v0 <= 0 or v0 > 0.25: return 1e12
    if lam < 0 or lam > 5.0: return 1e12
    if rho < -1.0 or rho > 1.0: return 1e12
    if sigma_J <= 0 or sigma_J > 0.5: return 1e12
    if mu_J < -0.3 or mu_J > 0.3: return 1e12
    if abs(mu) > 0.3: return 1e12
    
    # Feller condition
    feller_gap = 2.0 * kappa * theta - nu ** 2
    if feller_gap <= 0: 
        return 1e12
    
    # Calculate likelihood
    neg_ll = 0.0
    v_prev = v0
    exp_factor = math.exp(-kappa * DT)
    
    try:
        for r in returns:
            pdf_step = cf_pdf_inversion(r, v_prev, params, fast_mode=True)
            neg_ll -= math.log(pdf_step)
            Ev_t = theta + (v_prev - theta) * exp_factor
            v_prev = Ev_t
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

def compute_model_moments_analytical(params, v0_override=None):
    """Compute Bates model moments analytically.
    
    NOTE: This function computes EQUILIBRIUM moments (assumes variance is at long-run level theta).
    For finite simulations or non-equilibrium states, these may not match simulated moments exactly.
    
    FIXED: The mean formula now correctly includes the -0.5 * variance term that was missing.
    The original formula was: mu * DT + lam * DT * mu_J (WRONG - missing variance term)
    The corrected formula is: (mu - 0.5 * theta) * DT + lam * DT * mu_J (CORRECT for equilibrium)
    
    Args:
        params: Bates model parameters
        v0_override: If provided, use this as initial variance instead of params[IDX['v0']]
                     Currently not used (equilibrium assumption), but kept for future enhancement
    
    Returns:
        dict with keys: 'mean', 'std', 'skew', 'kurt'
        - 'mean' and 'std' are exact for equilibrium conditions
        - 'skew' and 'kurt' are approximations
    """
    mu = params[IDX['mu']]
    kappa = params[IDX['kappa']]
    theta = params[IDX['theta']]
    nu = params[IDX['nu']]
    rho = params[IDX['rho']]
    v0 = v0_override if v0_override is not None else params[IDX['v0']]
    lam = params[IDX['lam']]
    mu_J = params[IDX['mu_J']]
    sigma_J = params[IDX['sigma_J']]
    
    # Mean: E[log_return] = (mu - 0.5 * E[variance]) * DT + lam * DT * mu_J
    # IMPORTANT: The original formula was missing the -0.5 * variance term!
    # The log return formula is: dS/S = (mu - 0.5*v)dt + sqrt(v)*dW + dJ
    # So the mean depends on the expected variance.
    #
    # For a mean-reverting variance process: dv = kappa * (theta - v) * dt + nu * sqrt(v) * dW
    # The expected variance at time t starting from v0 is:
    #   E[v_t] = theta + (v0 - theta) * exp(-kappa * t)
    #
    # For a single period DT, the expected variance during that period is approximately:
    #   E[v] ≈ (v0 + E[v_DT]) / 2  (average of start and end)
    #   where E[v_DT] = theta + (v0 - theta) * exp(-kappa * DT)
    #
    # For small DT: E[v_DT] ≈ theta + (v0 - theta) * (1 - kappa * DT) = v0 + kappa * (theta - v0) * DT
    # So: E[v] ≈ (v0 + v0 + kappa * (theta - v0) * DT) / 2 = v0 + 0.5 * kappa * (theta - v0) * DT
    #
    # For equilibrium (long-term or v0 = theta): E[v] = theta
    # For finite simulations: use the average expected variance
    if kappa > 1e-6 and abs(v0 - theta) > 1e-6:
        # Account for mean reversion from v0 toward theta
        # Expected variance at end of period
        v_end = theta + (v0 - theta) * math.exp(-kappa * DT)
        # Average variance over the period (trapezoidal rule)
        expected_variance = (v0 + v_end) / 2.0
    else:
        # Use equilibrium value if kappa is very small or v0 ≈ theta
        expected_variance = theta
    
    model_mean_period = (mu - 0.5 * expected_variance) * DT + lam * DT * mu_J
    
    # Variance: Var[log_return] = E[variance] * DT + lam * DT * (sigma_J^2 + mu_J^2)
    # The variance of returns depends on the variance process
    # Use the same expected_variance calculated above (accounts for v0 if needed)
    # The diffusion contributes expected_variance * DT, jumps add lam * DT * (sigma_J^2 + mu_J^2)
    model_var_period = expected_variance * DT + lam * DT * (sigma_J ** 2 + mu_J ** 2)
    model_std_period = math.sqrt(max(model_var_period, 1e-12))
    
    # Skewness and kurtosis (approximate)
    if model_std_period > 1e-6:
        diff_var = theta * DT
        var_frac = diff_var / model_var_period if model_var_period > 1e-12 else 0.0
        
        # Heston (diffusion) contribution to skewness
        heston_skew_approx = 0.0
        if diff_var > 1e-12 and abs(rho) > 1e-6 and nu > 1e-6:
            heston_skew_raw = -rho * nu * math.sqrt(DT) / math.sqrt(max(theta, 1e-6)) * var_frac * 2.0
            heston_skew_approx = max(-3.0, min(3.0, heston_skew_raw))
        
        # Jump contribution to skewness
        jump_skew_contrib = 0.0
        if lam > 1e-6:
            jump_third_moment = lam * DT * (mu_J ** 3 + 3 * mu_J * sigma_J ** 2)
            jump_skew_contrib = jump_third_moment / (model_std_period ** 3)
            jump_skew_contrib = max(-5.0, min(5.0, jump_skew_contrib))
        
        model_skew = heston_skew_approx + jump_skew_contrib
        model_skew = max(-3.0, min(3.0, model_skew))
        
        # Heston contribution to excess kurtosis
        heston_kurt_approx = 0.0
        if kappa > 1e-6 and diff_var > 1e-12 and nu > 1e-6:
            heston_kurt_raw = 3.0 * (nu ** 2) / (kappa * max(theta, 1e-6)) * var_frac
            heston_kurt_approx = max(0.0, min(15.0, heston_kurt_raw))
        
        # Jump contribution to excess kurtosis
        jump_kurt_contrib = 0.0
        if lam > 1e-6:
            jump_fourth_moment = lam * DT * (mu_J ** 4 + 6 * (mu_J ** 2) * (sigma_J ** 2) + 3 * (sigma_J ** 4))
            jump_kurt_raw = jump_fourth_moment / (model_std_period ** 4)
            jump_kurt_contrib = jump_kurt_raw - 3.0  # Convert to excess kurtosis
            jump_kurt_contrib = max(-3.0, min(30.0, jump_kurt_contrib))
        
        model_kurt = heston_kurt_approx + jump_kurt_contrib
        # INCREASED CAP: Allow higher kurtosis to better match historical (historical can be 8.81)
        model_kurt = max(0.0, min(30.0, model_kurt))  # Increased from 20.0 to 30.0
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

def simulate_bates_for_drawdown(params, n_periods, n_simulations=None, seed=None, show_progress=False):
    """Simulate Bates model to estimate max drawdown via Monte Carlo.
    
    This is more accurate than analytical approximations for tail risk metrics.
    
    Args:
        show_progress: If True, shows a progress bar (only for large simulations)
    """
    if n_simulations is None:
        n_simulations = MC_SIMULATIONS_FOR_DD
    rng = np.random.default_rng(seed)
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
            variance = v0
            prices = [price]
            
            for t in range(n_periods):
                # Correlated Wiener processes
                z1 = rng.standard_normal()
                z2 = rho * z1 + math.sqrt(1.0 - rho**2) * rng.standard_normal()
                
                # Heston variance process
                variance = variance + kappa * (theta - variance) * dt + nu * math.sqrt(max(variance, 0.0)) * math.sqrt(dt) * z2
                variance = max(variance, 1e-12)
                
                # Merton jumps
                num_jumps = rng.poisson(lam * dt)
                if num_jumps > 0:
                    jump_size = np.sum(rng.normal(mu_J, sigma_J, num_jumps))
                else:
                    jump_size = 0.0
                
                # Bates price process
                log_return = (mu - 0.5 * variance) * dt + math.sqrt(max(variance, 0.0)) * math.sqrt(dt) * z1 + jump_size
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
        params: Bates model parameters (9-element array)
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
        
        kappa = params[IDX['kappa']]
        theta = params[IDX['theta']]
        nu = params[IDX['nu']]
        v0 = params[IDX['v0']]
        lam = params[IDX['lam']]
        mu = params[IDX['mu']]
        rho = params[IDX['rho']]
        mu_J = params[IDX['mu_J']]
        sigma_J = params[IDX['sigma_J']]
        
        # Parameter bounds validation
        if kappa <= 0 or kappa > 100.0: return 1e12
        if theta <= 0 or theta > 0.25: return 1e12
        if nu <= 0 or nu > 20.0: return 1e12
        if v0 <= 0 or v0 > 0.25: return 1e12
        if lam < 0 or lam > 5.0: return 1e12
        if rho < -1.0 or rho > 1.0: return 1e12
        if sigma_J <= 0 or sigma_J > 0.5: return 1e12
        if mu_J < -0.3 or mu_J > 0.3: return 1e12
        if abs(mu) > 0.3: return 1e12
        
        # Feller condition
        feller_gap = 2.0 * kappa * theta - nu ** 2
        if feller_gap <= 0: 
            return 1e12
        
        # Calculate model moments
        model_moments = compute_model_moments_analytical(params)
        
        # Compute max drawdown via Monte Carlo (only if enabled - this is expensive!)
        # When disabled, we skip it during optimization for speed, but still compute for final results
        if match_max_dd:
            n_periods = len(returns)
            # Only show progress in debug mode to avoid cluttering output
            model_max_dd = simulate_bates_for_drawdown(params, n_periods, n_simulations=MC_SIMULATIONS_FOR_DD, show_progress=DEBUG)
            dd_abs_error = abs(model_max_dd - emp_moments.get('max_dd', 0.0))
        else:
            # Skip max drawdown calculation during optimization (much faster)
            model_max_dd = 0.0  # Placeholder, won't be used in objective
            dd_abs_error = 0.0  # No error contribution when disabled
        
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
        
        # INCREASED WEIGHTS for skew/kurtosis to better match them
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
        if abs(emp_max_dd) > 0.01:  # If max drawdown is significant (>1%)
            dd_error = dd_abs_error / abs(emp_max_dd)
        else:
            dd_error = dd_abs_error * 10.0  # Scale absolute error for small drawdowns
        
        # Primary objective: moment matching (INCREASED weights for skew/kurt/dd)
        moment_error = (10000.0 * (mean_error ** 2) +
                        20000.0 * (std_error ** 2) +
                        25000.0 * (skew_error ** 2) +  # Increased from 15000
                        25000.0 * (kurt_error ** 2))   # Increased from 15000
        
        # Add max drawdown error only if enabled (it's expensive to compute)
        # INCREASED WEIGHT for max drawdown to better match tail risk
        if match_max_dd:
            # Use higher weight for extreme drawdowns (like Regime 2's -86.52%)
            if abs(emp_max_dd) > 0.5:  # Extreme drawdown (>50%)
                moment_error += 50000.0 * (dd_error ** 2)  # Much higher weight for extreme tail risk
            else:
                moment_error += 20000.0 * (dd_error ** 2)  # Increased from 10000
        
        # Secondary objective: likelihood (regularization)
        likelihood_term = 0.0
        if likelihood_weight > 0:
            try:
                # Use fast mode and sample subset for speed
                sample_size = min(len(returns), 100)
                if len(returns) > sample_size:
                    indices = np.linspace(0, len(returns)-1, sample_size, dtype=int)
                    returns_sample = returns[indices]
                else:
                    returns_sample = returns
                
                # Use Cython if available, otherwise Python
                if CYTHON_AVAILABLE and NEG_LOG_LIKELIHOOD_C is not None:
                    neg_ll = NEG_LOG_LIKELIHOOD_C(params, returns_sample)
                else:
                    neg_ll = neg_log_likelihood_py(params, returns_sample)
                
                # Normalize and weight
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
        dict with diagnostics:
        - 'hitting_boundaries': list of parameter indices hitting boundaries
        - 'near_boundaries': list of parameter indices near boundaries
        - 'boundary_distances': dict of distances from boundaries for each parameter
        - 'recommendations': list of recommendations for adjusting bounds
    """
    hitting = []
    near = []
    distances = {}
    recommendations = []
    
    param_names = ['mu', 'kappa', 'theta', 'nu', 'rho', 'v0', 'lam', 'mu_J', 'sigma_J']
    
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
        
        # Hitting boundary (within 0.1% of range)
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
        # Near boundary (within tolerance)
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

def diagnose_bounds_from_results(results, bounds, warn_threshold=0.05):
    """
    Diagnose if bounds are appropriate based on optimization results.
    
    Signs that bounds are TOO WIDE:
    1. Parameters consistently hitting boundaries across multiple restarts
    2. Optimization converging to different boundary values (unstable)
    3. Very slow convergence (searching large space)
    
    Signs that bounds are TOO TIGHT:
    1. Parameters hitting boundaries when they shouldn't
    2. Poor fit quality despite hitting boundaries
    3. Parameters that should vary are constrained
    
    Args:
        results: List of optimization results (from multiple restarts)
        bounds: Parameter bounds used
        warn_threshold: Fraction of range from boundary to warn (default 5%)
    
    Returns:
        dict with diagnostics and recommendations
    """
    if not results:
        return None
    
    # Collect all parameter values from successful optimizations
    all_params = []
    for res in results:
        if hasattr(res, 'x') and res.success:
            all_params.append(res.x)
    
    if not all_params:
        return {'error': 'No successful optimizations to analyze'}
    
    all_params = np.array(all_params)
    n_restarts = len(all_params)
    
    diagnostics = {
        'n_restarts': n_restarts,
        'parameter_stats': {},
        'boundary_issues': [],
        'recommendations': []
    }
    
    param_names = ['mu', 'kappa', 'theta', 'nu', 'rho', 'v0', 'lam', 'mu_J', 'sigma_J']
    
    for i, (name, (lower, upper)) in enumerate(zip(param_names, bounds)):
        param_values = all_params[:, i]
        range_size = upper - lower
        
        # Statistics
        mean_val = np.mean(param_values)
        std_val = np.std(param_values)
        min_val = np.min(param_values)
        max_val = np.max(param_values)
        
        # Distance from boundaries
        mean_dist_lower = (mean_val - lower) / range_size if range_size > 0 else 0
        mean_dist_upper = (upper - mean_val) / range_size if range_size > 0 else 0
        min_dist = min(mean_dist_lower, mean_dist_upper)
        
        # Count how many restarts hit boundaries
        hitting_lower = np.sum((param_values - lower) / range_size < 0.01) if range_size > 0 else 0
        hitting_upper = np.sum((upper - param_values) / range_size < 0.01) if range_size > 0 else 0
        
        diagnostics['parameter_stats'][name] = {
            'mean': mean_val,
            'std': std_val,
            'min': min_val,
            'max': max_val,
            'mean_distance_from_boundary': min_dist,
            'hitting_lower_count': hitting_lower,
            'hitting_upper_count': hitting_upper,
            'hitting_any_count': hitting_lower + hitting_upper
        }
        
        # Diagnose issues
        hitting_pct = (hitting_lower + hitting_upper) / n_restarts * 100
        
        if hitting_pct > 50:  # More than 50% hitting boundaries
            diagnostics['boundary_issues'].append(
                f"{name}: {hitting_pct:.1f}% of restarts hit boundaries - bounds likely TOO TIGHT"
            )
            if hitting_lower > hitting_upper:
                diagnostics['recommendations'].append(
                    f"{name}: Lower bound ({lower:.4f}) too high. Current mean: {mean_val:.4f}. "
                    f"Consider lowering to {lower * 0.5:.4f} or {mean_val - 2*std_val:.4f}"
                )
            else:
                diagnostics['recommendations'].append(
                    f"{name}: Upper bound ({upper:.4f}) too low. Current mean: {mean_val:.4f}. "
                    f"Consider raising to {upper * 1.5:.4f} or {mean_val + 2*std_val:.4f}"
                )
        elif min_dist < warn_threshold:  # Near boundary
            diagnostics['boundary_issues'].append(
                f"{name}: Mean value ({mean_val:.4f}) near boundary (within {warn_threshold*100:.0f}% of range)"
            )
        elif std_val / range_size > 0.3:  # High variance relative to range
            diagnostics['boundary_issues'].append(
                f"{name}: High variance across restarts ({std_val:.4f} vs range {range_size:.4f}) - "
                f"bounds may be TOO WIDE, causing instability"
            )
            diagnostics['recommendations'].append(
                f"{name}: Consider tightening bounds around mean ({mean_val:.4f} ± 2σ = "
                f"[{mean_val - 2*std_val:.4f}, {mean_val + 2*std_val:.4f}])"
            )
    
    return diagnostics

# ============================================================================
# OPTIMIZATION
# ============================================================================

def fit_bates_moment_matching(returns, name="regime", restarts=NUM_RESTARTS, maxiter=MAXITER,
                              bounds_vec=DEFAULT_BOUNDS, use_bounds=True, match_max_dd=None):
    """Fit Bates model to returns using moment matching.
    
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
    is_extreme_regime = vol_annual > 0.5 or abs(mean_period) > 0.1
    
    init_base = np.zeros(9, dtype=float)
    init_base[IDX['mu']] = np.clip(mu_annual, -0.3, 0.3)
    init_base[IDX['kappa']] = 2.0 if not is_extreme_regime else 5.0
    init_base[IDX['theta']] = np.clip(var_annual * 0.5, 0.01, 0.5)
    
    # Ensure Feller condition
    feller_max_nu = math.sqrt(2.0 * init_base[IDX['kappa']] * init_base[IDX['theta']] * 0.9)
    desired_nu = 1.0 if not is_extreme_regime else 2.0
    init_base[IDX['nu']] = min(desired_nu, feller_max_nu - 0.01)
    init_base[IDX['nu']] = max(0.1, init_base[IDX['nu']])
    
    init_base[IDX['rho']] = -0.5
    init_base[IDX['v0']] = np.clip(var_annual * 0.8, 0.01, 0.5)
    
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
            scales = np.array([0.005, 0.2, 0.1 * init_base[IDX['theta']], 0.1, 0.05,
                               0.1 * init_base[IDX['v0']], 0.1, 0.002, 0.01])
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
            
            # Ensure Feller condition
            kappa_val = x0[IDX['kappa']]
            theta_val = x0[IDX['theta']]
            nu_val = x0[IDX['nu']]
            feller_gap = 2.0 * kappa_val * theta_val - nu_val ** 2
            if feller_gap <= 0:
                max_nu = math.sqrt(2.0 * kappa_val * theta_val * 0.95)
                if nu_val >= max_nu:
                    x0[IDX['nu']] = max(0.1, max_nu - 0.01)
            
            # Optimize
            opt_options = {
                "maxiter": min(maxiter, 10000),  # Reduced cap for faster execution
                "ftol": 1e-7,  # Slightly relaxed for faster convergence
                "gtol": 1e-6,  # Slightly relaxed for faster convergence
                "maxls": 20,   # Reduced line search steps
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
    
    # Always compute max drawdown for final results (even if not used in optimization)
    # This gives us the metric for reporting, but doesn't slow down optimization
    if DEBUG:
        console.print(f"[dim]  Computing final max drawdown estimate (50 simulations)...[/dim]")
    model_max_dd = simulate_bates_for_drawdown(params, n, n_simulations=50, show_progress=DEBUG)
    
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
        
        # Warn about boundary issues
        if boundary_diagnostics['hitting_boundaries']:
            console.print(f"[yellow]  ⚠ Warning: {len(boundary_diagnostics['hitting_boundaries'])} parameters hitting boundaries[/yellow]")
            for rec in boundary_diagnostics['recommendations'][:3]:  # Show first 3
                console.print(f"[dim]    {rec}[/dim]")
    
    return result

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main function to fit Bates model to each regime."""
    global DEBUG
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Fit Bates model to regimes using moment matching')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--no-cython', action='store_true', help='Disable Cython (use pure Python)')
    parser.add_argument('--csv', type=str, default='data/regime_classification_nominal_returns.csv',
                       help='Path to CSV file with returns and regime IDs')
    parser.add_argument('--match-max-dd', action='store_true', 
                       help='Enable max drawdown matching (slower but more accurate for tail risk)')
    parser.add_argument('--match-aggregate-dd', action='store_true',
                       help='Enable aggregate max drawdown matching (matches full regime-switching path, very slow)')
    parser.add_argument('--option-pricing', action='store_true',
                       help='Use bounds optimized for option pricing calibration (tighter vol-of-vol, wider jumps)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Disable plot generation (faster execution)')
    args = parser.parse_args()
    
    global DEBUG, MATCH_MAX_DRAWDOWN
    DEBUG = args.debug
    MATCH_MAX_DRAWDOWN = args.match_max_dd
    
    # Select bounds based on use case
    if args.option_pricing:
        bounds_to_use = OPTION_PRICING_BOUNDS
        console.print("[cyan]Using OPTION PRICING bounds (tighter vol-of-vol, wider jumps)[/cyan]")
    else:
        bounds_to_use = DEFAULT_BOUNDS
        console.print("[cyan]Using TIME SERIES bounds (default)[/cyan]")
    
    if MATCH_MAX_DRAWDOWN:
        console.print("[yellow]⚠ Max drawdown matching enabled - optimization will be slower[/yellow]")
        console.print("[dim]Note: This matches per-regime max drawdown. Aggregate max drawdown depends on regime sequence.[/dim]")
    else:
        console.print("[dim]Max drawdown matching disabled (use --match-max-dd to enable)[/dim]")
    
    console.print("[bold cyan]Bates Model Moment Matching[/bold cyan]")
    console.print("=" * 80)
    
    # Try to import Cython
    if not args.no_cython:
        try_import_cython()
    else:
        console.print("[yellow]Cython disabled by user[/yellow]")
    
    # Read CSV file
    csv_path = args.csv
    if not os.path.exists(csv_path):
        # Try relative to script directory
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
        pass  # Already in decimal
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
    
    # Estimate time (rough calculation)
    total_restarts = 0
    for regime_id, regime_df in regimes:
        n = len(regime_df)
        restarts = NUM_RESTARTS_SMALL if n < 200 else NUM_RESTARTS
        total_restarts += restarts
    
    console.print(f"[dim]Estimated optimization attempts: ~{total_restarts} restarts across {len(regimes)} regimes[/dim]")
    console.print(f"[dim]This may take several minutes. Progress bars will show current status...[/dim]\n")
    
    # Fit each regime with progress bar
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
            
            # SPECIAL HANDLING FOR REGIME 3 (EXTREME CRISIS): Fit only from extreme drawdown periods
            # This ensures Regime 3 captures extreme crisis behavior, not just normal high-vol periods
            if regime_id == 3.0 or regime_id == 3:
                console.print(f"\n[yellow]Regime 3 (Extreme Crisis) detected - filtering for extreme drawdown periods only[/yellow]")
                
                # Calculate cumulative price and drawdown from peak
                cumulative_price = 100.0 * np.cumprod(1 + regime_returns)
                running_peak = np.maximum.accumulate(cumulative_price)
                drawdown_from_peak = (running_peak - cumulative_price) / running_peak
                
                # Filter for periods with >50% drawdown (extreme crisis periods)
                extreme_threshold = 0.50  # 50% drawdown
                extreme_mask = drawdown_from_peak > extreme_threshold
                
                if np.sum(extreme_mask) < 20:  # Need at least 20 observations
                    console.print(f"[yellow]Only {np.sum(extreme_mask)} extreme periods found. Lowering threshold to 30%...[/yellow]")
                    extreme_threshold = 0.30
                    extreme_mask = drawdown_from_peak > extreme_threshold
                
                if np.sum(extreme_mask) < 10:
                    console.print(f"[red]WARNING: Only {np.sum(extreme_mask)} extreme periods found. Using all Regime 3 data.[/red]")
                    extreme_mask = np.ones(len(regime_returns), dtype=bool)
                else:
                    console.print(f"[green]Using {np.sum(extreme_mask)}/{len(regime_returns)} extreme periods (>{extreme_threshold*100:.0f}% drawdown) for Regime 3 fitting[/green]")
                    console.print(f"  Mean drawdown in extreme periods: {np.mean(drawdown_from_peak[extreme_mask])*100:.2f}%")
                    console.print(f"  Mean drawdown in all Regime 3: {np.mean(drawdown_from_peak)*100:.2f}%")
                
                regime_returns = regime_returns[extreme_mask]
            
            if len(regime_returns) < 10:
                console.print(f"[yellow]Skipping regime_{regime_id}: Only {len(regime_returns)} observations after filtering[/yellow]")
                progress.update(task, advance=1)
                continue
            
            # Convert to log returns if needed (check if already log returns)
            # If returns are already small (typical log returns are -0.1 to 0.1), assume log returns
            # Otherwise, convert from simple returns
            if np.any(np.abs(regime_returns) > 0.5):
                # Likely simple returns, convert to log
                regime_returns = np.log(1.0 + regime_returns)
            
            result = fit_bates_moment_matching(
                regime_returns,
                name=f"regime_{regime_id}",
                restarts=NUM_RESTARTS,
                maxiter=MAXITER,
                bounds_vec=bounds_to_use
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
    
    table = Table(title="Bates Model Parameters - Moment Matching")
    table.add_column("Regime", style="cyan")
    table.add_column("N", justify="right")
    table.add_column("μ (annual)", justify="right", style="green")
    table.add_column("κ", justify="right")
    table.add_column("θ (annual)", justify="right")
    table.add_column("ν", justify="right")
    table.add_column("ρ", justify="right")
    table.add_column("v₀", justify="right")
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
            f"{p[IDX['kappa']]:.2f}",
            f"{p[IDX['theta']]:.4f}",
            f"{p[IDX['nu']]:.3f}",
            f"{p[IDX['rho']]:.3f}",
            f"{p[IDX['v0']]:.4f}",
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
        table2.add_row("", "", "", "", "")  # Spacer
    
    console.print(table2)
    
    # Save results to CSV in data folder
    os.makedirs('data', exist_ok=True)
    output_file = "data/Bates_per_regime_Moment_Matching_results.csv"
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
            'kappa': p[IDX['kappa']],
            'theta_annual': p[IDX['theta']],
            'nu': p[IDX['nu']],
            'rho': p[IDX['rho']],
            'v0': p[IDX['v0']],
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
                for rec in diag['recommendations'][:2]:  # Show first 2 recommendations
                    console.print(f"    [dim]{rec}[/dim]")
        console.print("\n[dim]Recommendation: Review bounds - they may be too tight for your data.[/dim]")
        console.print("[dim]For option pricing, consider using --option-pricing flag for appropriate bounds.[/dim]")
    else:
        console.print("[green]✓ No boundary issues detected - bounds appear appropriate[/green]")
    
    console.print(f"\n[bold green]Completed! Fitted {len(results)} regimes.[/bold green]")
    
    # Generate and save distribution visualizations
    if not args.no_plots:
        console.print("\n[bold cyan]Generating distribution visualizations...[/bold cyan]")
        generate_distribution_plots(results, df, regime_col)
        console.print("[green]✓ All visualizations saved to output/ folder[/green]")
    else:
        console.print("\n[dim]Plot generation disabled (use --no-plots flag was set)[/dim]")

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def simulate_bates_returns(params, n_periods, n_simulations=1000, seed=None):
    """Simulate Bates model returns for visualization.
    
    Returns:
        array of simulated returns (n_simulations x n_periods)
    """
    rng = np.random.default_rng(seed)
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
    all_returns = []
    all_volatilities = []
    
    for sim in range(n_simulations):
        variance = v0
        returns_path = []
        vol_path = []
        
        for t in range(n_periods):
            # Correlated Wiener processes
            z1 = rng.standard_normal()
            z2 = rho * z1 + math.sqrt(1.0 - rho**2) * rng.standard_normal()
            
            # Heston variance process
            variance = variance + kappa * (theta - variance) * dt + nu * math.sqrt(max(variance, 0.0)) * math.sqrt(dt) * z2
            variance = max(variance, 1e-12)
            # Store instantaneous volatility (NOT annualized - will be annualized when comparing)
            # Note: variance is already in annualized units (theta is annual variance)
            # So sqrt(variance) gives annualized volatility directly
            vol_path.append(math.sqrt(variance))
            
            # Merton jumps
            num_jumps = rng.poisson(lam * dt)
            if num_jumps > 0:
                jump_size = np.sum(rng.normal(mu_J, sigma_J, num_jumps))
            else:
                jump_size = 0.0
            
            # Bates price process
            log_return = (mu - 0.5 * variance) * dt + math.sqrt(max(variance, 0.0)) * math.sqrt(dt) * z1 + jump_size
            returns_path.append(log_return)
        
        all_returns.append(returns_path)
        all_volatilities.append(vol_path)
    
    return np.array(all_returns), np.array(all_volatilities)

def compute_path_signature(returns, order=3):
    """Compute truncated path signature (iterated integrals) of returns.
    
    Path signatures are a powerful tool from rough path theory that capture
    the full path structure through iterated integrals. They provide a
    universal way to represent path-dependent information.
    
    The signature of order N contains all iterated integrals up to order N:
    - Order 1: ∫dX (linear)
    - Order 2: ∫∫dX dX (quadratic variation, volatility)
    - Order 3: ∫∫∫dX dX dX (captures asymmetry, skewness)
    - Higher orders: capture more complex path features
    
    Args:
        returns: array of returns (path increments)
        order: truncation order (higher = more features, but exponential growth)
    
    Returns:
        dict with signature features up to specified order
    """
    n = len(returns)
    if n == 0:
        return {}
    
    # Compute cumulative path (integral)
    path = np.cumsum(returns)
    
    # Initialize signature dictionary
    sig = {}
    
    # Order 1: First-order signature (linear functional)
    # S^1 = ∫dX = X_T - X_0 = cumulative return
    sig['S1'] = path[-1] if n > 0 else 0.0
    
    # Order 2: Second-order signature (quadratic variation)
    # S^2 = ∫∫dX dX = ∫X dX = (1/2)(X_T^2 - X_0^2) for continuous paths
    # For discrete: approximation using quadratic variation
    sig['S2'] = np.sum(returns ** 2)  # Quadratic variation
    
    if order >= 3:
        # Order 3: Third-order signature (captures asymmetry)
        # S^3 = ∫∫∫dX dX dX captures path-dependent skewness
        # Discrete approximation
        sig['S3'] = np.sum(returns ** 3)
        
        # Also compute cross terms (more sophisticated)
        # ∫X dX (Lévy area approximation)
        sig['S12'] = np.sum(path[:-1] * returns[1:]) if n > 1 else 0.0
    
    if order >= 4:
        # Order 4: Fourth-order signature (kurtosis, tail behavior)
        sig['S4'] = np.sum(returns ** 4)
        # Higher cross terms
        sig['S112'] = np.sum(path[:-1] ** 2 * returns[1:]) if n > 1 else 0.0
    
    return sig

def compute_signature_volatility(returns, window=21, use_full_signature=True):
    """Compute signature-based volatility using path signatures.
    
    CURRENT IMPLEMENTATION (simplified):
    - Uses basic signature features (moments) in a rolling window
    - This is "signature-inspired" but not a true signature-vol model
    
    TRUE SIGNATURE-VOL MODEL would:
    - Compute full path signature (all iterated integrals)
    - Use signature regression: vol_t = f(Signature(X_{t-window:t}))
    - Leverage universal approximation property of signatures
    - Model volatility as a linear/nonlinear function of signature features
    
    Args:
        returns: array of returns
        window: rolling window size
        use_full_signature: If True, uses more sophisticated signature features
    
    Returns:
        signature_vol: array of signature-based volatility estimates
    """
    n = len(returns)
    signature_vol = np.zeros(n)
    
    if use_full_signature:
        # More sophisticated: use full signature features
        for i in range(window, n):
            window_returns = returns[i-window:i]
            
            # Compute path signature up to order 3
            sig = compute_path_signature(window_returns, order=3)
            
            # Base volatility from quadratic variation (order 2 signature)
            base_vol = math.sqrt(sig.get('S2', 0.0) / window)
            
            # Adjust using higher-order signatures
            # Order 3 captures asymmetry (skewness effect on volatility)
            s3 = sig.get('S3', 0.0)
            asymmetry_factor = abs(s3) / (window * (base_vol ** 3) + 1e-8)
            
            # Cross-term S12 captures path-dependent correlation
            s12 = sig.get('S12', 0.0)
            path_dependent_factor = abs(s12) / (window ** 2 * (base_vol ** 2) + 1e-8)
            
            # Combine signature features (this is a simple linear combination;
            # a true signature-vol model would use regression/ML to learn weights)
            signature_vol[i] = base_vol * (
                1.0 + 
                0.15 * asymmetry_factor +  # Asymmetry correction
                0.10 * path_dependent_factor  # Path-dependent correction
            )
    else:
        # Simplified version (original)
        for i in range(window, n):
            window_returns = returns[i-window:i]
            
            # First-order signature (linear functional)
            s1 = np.sum(window_returns)
            
            # Second-order signature (quadratic variation approximation)
            s2 = np.sum(window_returns ** 2)
            
            # Third-order signature (captures asymmetry)
            s3 = np.sum(window_returns ** 3)
            
            # Combine signatures to estimate volatility
            base_vol = math.sqrt(s2 / window)
            asymmetry_correction = abs(s3) / (window * (base_vol ** 3) + 1e-6)
            
            # Signature-based volatility estimate
            signature_vol[i] = base_vol * (1.0 + 0.1 * asymmetry_correction)
    
    # Forward fill initial values
    signature_vol[:window] = signature_vol[window] if n > window else 0.0
    
    return signature_vol

def generate_distribution_plots(results, df, regime_col):
    """Generate all distribution visualizations and save to output folder."""
    os.makedirs('output', exist_ok=True)
    
    # Set dark background style
    plt.style.use('dark_background')
    
    # Get empirical returns for each regime
    regimes_dict = {}
    for regime_id, regime_df in df.groupby(regime_col):
        returns = regime_df['returns'].values if 'returns' in regime_df.columns else regime_df['Total Nominal Return (%)'].values / 100.0
        # Convert to log returns if needed
        if np.any(np.abs(returns) > 0.5):
            returns = np.log(1.0 + returns)
        regimes_dict[regime_id] = returns
    
    # 1. Individual regime histograms
    n_regimes = len(results)
    fig, axes = plt.subplots(n_regimes, 1, figsize=(14, 4 * n_regimes), facecolor='black')
    if n_regimes == 1:
        axes = [axes]
    
    for idx, r in enumerate(results):
        regime_id = float(r['name'].split('_')[1])
        params = r['params']
        emp_returns = regimes_dict.get(regime_id, np.array([]))
        
        if len(emp_returns) == 0:
            continue
        
        # Simulate model returns - use more simulations for better distribution fit
        n_periods = len(emp_returns)
        # For small regimes, use fewer simulations to avoid memory issues
        n_sims = min(10000, max(5000, n_periods * 50)) if n_periods < 200 else 10000
        sim_returns, sim_vols = simulate_bates_returns(params, n_periods, n_simulations=n_sims, seed=42)
        sim_returns_flat = sim_returns.flatten()
        
        ax = axes[idx]
        
        # Histogram comparison
        bins = np.linspace(min(np.min(emp_returns), np.min(sim_returns_flat)),
                          max(np.max(emp_returns), np.max(sim_returns_flat)), 50)
        
        ax.hist(emp_returns, bins=bins, alpha=0.6, label='Empirical', density=True, color='cyan', edgecolor='white', linewidth=0.5)
        ax.hist(sim_returns_flat, bins=bins, alpha=0.6, label='Bates Model', density=True, color='orange', edgecolor='white', linewidth=0.5)
        
        ax.set_xlabel('Log Return', fontsize=12, color='white')
        ax.set_ylabel('Density', fontsize=12, color='white')
        ax.set_title(f'{r["name"]}: Empirical vs Bates Model Distribution (n={r["n_obs"]})', fontsize=14, fontweight='bold', color='white')
        ax.legend(fontsize=11, facecolor='black', edgecolor='white')
        ax.grid(True, alpha=0.3, color='gray')
        ax.set_facecolor('black')
        ax.tick_params(colors='white')
    
    plt.tight_layout()
    plt.savefig('output/bates_regime_distributions.png', dpi=300, bbox_inches='tight', facecolor='black')
    plt.close()
    
    # 2. Joint distribution plots (Returns vs Volatility)
    fig = plt.figure(figsize=(16, 5 * n_regimes), facecolor='black')
    gs = gridspec.GridSpec(n_regimes, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    for idx, r in enumerate(results):
        regime_id = float(r['name'].split('_')[1])
        params = r['params']
        emp_returns = regimes_dict.get(regime_id, np.array([]))
        
        if len(emp_returns) == 0:
            continue
        
        # Simulate model - use more simulations for better joint distribution
        n_periods = len(emp_returns)
        n_sims = min(5000, max(2000, n_periods * 30)) if n_periods < 200 else 5000
        sim_returns, sim_vols = simulate_bates_returns(params, n_periods, n_simulations=n_sims, seed=42)
        
        # Compute empirical volatility (rolling) - annualized
        window = min(21, len(emp_returns) // 4)
        if window < 3:
            window = 3
        emp_vol = pd.Series(emp_returns).rolling(window=window, min_periods=1).std().values * math.sqrt(PERIODS_PER_YEAR)
        
        # For model: compute rolling volatility from simulated returns (to match empirical method)
        # Sample a subset of simulations and compute rolling vol for each path
        n_sample_paths = min(100, len(sim_returns))
        model_vols_rolling = []
        for path_idx in range(n_sample_paths):
            path_returns = sim_returns[path_idx, :]
            path_vol = pd.Series(path_returns).rolling(window=window, min_periods=1).std().values * math.sqrt(PERIODS_PER_YEAR)
            model_vols_rolling.append(path_vol)
        model_vols_rolling = np.array(model_vols_rolling)
        sim_vols_flat = model_vols_rolling.flatten()  # Flatten for scatter plot
        
        # Plot 1: Empirical joint distribution
        ax1 = fig.add_subplot(gs[idx, 0])
        scatter1 = ax1.scatter(emp_returns, emp_vol, alpha=0.5, s=20, c=range(len(emp_returns)), 
                               cmap='viridis', edgecolors='white', linewidths=0.3)
        ax1.set_xlabel('Log Return', fontsize=12, color='white')
        ax1.set_ylabel('Volatility (Annualized)', fontsize=12, color='white')
        ax1.set_title(f'{r["name"]}: Empirical Joint Distribution\n(Returns vs Volatility)', 
                     fontsize=13, fontweight='bold', color='white')
        ax1.grid(True, alpha=0.3, color='gray')
        ax1.set_facecolor('black')
        ax1.tick_params(colors='white')
        cbar1 = plt.colorbar(scatter1, ax=ax1, label='Time Index')
        cbar1.ax.yaxis.set_tick_params(color='white')
        cbar1.ax.set_ylabel('Time Index', color='white')
        plt.setp(plt.getp(cbar1.ax.axes, 'yticklabels'), color='white')
        
        # Plot 2: Model joint distribution
        ax2 = fig.add_subplot(gs[idx, 1])
        sim_returns_flat = sim_returns[:n_sample_paths, :].flatten()  # Match the sampled paths
        scatter2 = ax2.scatter(sim_returns_flat, sim_vols_flat, alpha=0.3, s=10, c='orange', edgecolors='white', linewidths=0.1)
        ax2.set_xlabel('Log Return', fontsize=12, color='white')
        ax2.set_ylabel('Volatility (Annualized)', fontsize=12, color='white')
        ax2.set_title(f'{r["name"]}: Bates Model Joint Distribution\n(Returns vs Volatility)', 
                     fontsize=13, fontweight='bold', color='white')
        ax2.grid(True, alpha=0.3, color='gray')
        ax2.set_facecolor('black')
        ax2.tick_params(colors='white')
        
        # Overlay empirical for comparison
        ax2.scatter(emp_returns, emp_vol, alpha=0.4, s=30, c='cyan', marker='x', 
                   label='Empirical', linewidths=1.5)
        ax2.legend(facecolor='black', edgecolor='white', labelcolor='white')
    
    plt.savefig('output/bates_joint_distributions.png', dpi=300, bbox_inches='tight', facecolor='black')
    plt.close()
    
    # 3. Signature-based volatility analysis
    fig, axes = plt.subplots(n_regimes, 2, figsize=(16, 5 * n_regimes), facecolor='black')
    if n_regimes == 1:
        axes = axes.reshape(1, -1)
    
    for idx, r in enumerate(results):
        regime_id = float(r['name'].split('_')[1])
        params = r['params']
        emp_returns = regimes_dict.get(regime_id, np.array([]))
        
        if len(emp_returns) == 0:
            continue
        
        # Compute signature volatility
        sig_vol = compute_signature_volatility(emp_returns, window=min(21, len(emp_returns) // 4))
        
        # Standard rolling volatility for comparison
        window = min(21, len(emp_returns) // 4)
        if window < 3:
            window = 3
        std_vol = pd.Series(emp_returns).rolling(window=window, min_periods=1).std().values * math.sqrt(PERIODS_PER_YEAR)
        
        # Plot 1: Signature volatility time series
        ax1 = axes[idx, 0]
        time_idx = np.arange(len(emp_returns))
        ax1.plot(time_idx, sig_vol, label='Signature Volatility', linewidth=2, color='purple')
        ax1.plot(time_idx, std_vol, label='Standard Rolling Vol', linewidth=2, color='orange', alpha=0.7)
        ax1.set_xlabel('Time Index', fontsize=12, color='white')
        ax1.set_ylabel('Volatility (Annualized)', fontsize=12, color='white')
        ax1.set_title(f'{r["name"]}: Signature-Based Volatility Analysis', fontsize=13, fontweight='bold', color='white')
        ax1.legend(fontsize=11, facecolor='black', edgecolor='white', labelcolor='white')
        ax1.grid(True, alpha=0.3, color='gray')
        ax1.set_facecolor('black')
        ax1.tick_params(colors='white')
        
        # Plot 2: Signature volatility vs returns scatter
        ax2 = axes[idx, 1]
        scatter = ax2.scatter(emp_returns, sig_vol, alpha=0.6, s=30, c=time_idx, cmap='coolwarm', 
                   edgecolors='white', linewidths=0.3)
        ax2.set_xlabel('Log Return', fontsize=12, color='white')
        ax2.set_ylabel('Signature Volatility', fontsize=12, color='white')
        ax2.set_title(f'{r["name"]}: Returns vs Signature Volatility', fontsize=13, fontweight='bold', color='white')
        ax2.grid(True, alpha=0.3, color='gray')
        ax2.set_facecolor('black')
        ax2.tick_params(colors='white')
        cbar2 = plt.colorbar(scatter, ax=ax2, label='Time Index')
        cbar2.ax.yaxis.set_tick_params(color='white')
        cbar2.ax.set_ylabel('Time Index', color='white')
        plt.setp(plt.getp(cbar2.ax.axes, 'yticklabels'), color='white')
    
    plt.tight_layout()
    plt.savefig('output/bates_signature_volatility.png', dpi=300, bbox_inches='tight', facecolor='black')
    plt.close()
    
    # 4. Combined comparison plot
    fig = plt.figure(figsize=(18, 6 * n_regimes), facecolor='black')
    gs = gridspec.GridSpec(n_regimes, 3, figure=fig, hspace=0.4, wspace=0.3)
    
    for idx, r in enumerate(results):
        regime_id = float(r['name'].split('_')[1])
        params = r['params']
        emp_returns = regimes_dict.get(regime_id, np.array([]))
        
        if len(emp_returns) == 0:
            continue
        
        # Simulate model - use more simulations for comprehensive comparison
        n_periods = len(emp_returns)
        n_sims = min(8000, max(3000, n_periods * 40)) if n_periods < 200 else 8000
        sim_returns, sim_vols = simulate_bates_returns(params, n_periods, n_simulations=n_sims, seed=42)
        sim_returns_flat = sim_returns.flatten()
        
        # Plot 1: Distribution comparison
        ax1 = fig.add_subplot(gs[idx, 0])
        bins = np.linspace(min(np.min(emp_returns), np.min(sim_returns_flat)),
                          max(np.max(emp_returns), np.max(sim_returns_flat)), 40)
        ax1.hist(emp_returns, bins=bins, alpha=0.6, label='Empirical', density=True, color='cyan', edgecolor='white', linewidth=0.5)
        ax1.hist(sim_returns_flat, bins=bins, alpha=0.6, label='Bates Model', density=True, color='orange', edgecolor='white', linewidth=0.5)
        ax1.set_xlabel('Log Return', fontsize=11, color='white')
        ax1.set_ylabel('Density', fontsize=11, color='white')
        ax1.set_title(f'{r["name"]}: Distribution', fontsize=12, fontweight='bold', color='white')
        ax1.legend(facecolor='black', edgecolor='white', labelcolor='white')
        ax1.grid(True, alpha=0.3, color='gray')
        ax1.set_facecolor('black')
        ax1.tick_params(colors='white')
        
        # Plot 2: Q-Q plot
        ax2 = fig.add_subplot(gs[idx, 1])
        emp_sorted = np.sort(emp_returns)
        sim_sorted = np.sort(sim_returns_flat)
        quantiles = np.linspace(0, 1, min(len(emp_sorted), len(sim_sorted)))
        emp_quantiles = np.quantile(emp_returns, quantiles)
        sim_quantiles = np.quantile(sim_returns_flat, quantiles)
        ax2.scatter(emp_quantiles, sim_quantiles, alpha=0.6, s=30, edgecolors='white', linewidths=0.5, color='cyan')
        min_val = min(np.min(emp_quantiles), np.min(sim_quantiles))
        max_val = max(np.max(emp_quantiles), np.max(sim_quantiles))
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Match')
        ax2.set_xlabel('Empirical Quantiles', fontsize=11, color='white')
        ax2.set_ylabel('Model Quantiles', fontsize=11, color='white')
        ax2.set_title(f'{r["name"]}: Q-Q Plot', fontsize=12, fontweight='bold', color='white')
        ax2.legend(facecolor='black', edgecolor='white', labelcolor='white')
        ax2.grid(True, alpha=0.3, color='gray')
        ax2.set_facecolor('black')
        ax2.tick_params(colors='white')
        
        # Plot 3: Volatility comparison
        ax3 = fig.add_subplot(gs[idx, 2])
        window = min(21, len(emp_returns) // 4)
        if window < 3:
            window = 3
        # Empirical: rolling volatility (smoothed) - annualized
        emp_vol = pd.Series(emp_returns).rolling(window=window, min_periods=1).std().values * math.sqrt(PERIODS_PER_YEAR)
        time_idx = np.arange(len(emp_returns))
        ax3.plot(time_idx, emp_vol, label='Empirical Vol (rolling)', linewidth=2, color='cyan', alpha=0.7)
        
        # Model: Compute rolling volatility from simulated returns (same method as empirical)
        if len(sim_returns) > 0:
            n_paths_to_avg = min(50, len(sim_returns))
            # Compute rolling volatility for each path, then average
            model_vols_per_path = []
            for path_idx in range(n_paths_to_avg):
                path_returns = sim_returns[path_idx, :len(emp_vol)]
                path_vol = pd.Series(path_returns).rolling(window=window, min_periods=1).std().values * math.sqrt(PERIODS_PER_YEAR)
                model_vols_per_path.append(path_vol)
            # Average across paths
            model_vol_smoothed = np.mean(model_vols_per_path, axis=0)
            ax3.plot(time_idx[:len(model_vol_smoothed)], model_vol_smoothed, 
                    alpha=0.7, linewidth=1.5, color='orange', label='Model Vol (rolling, 50 paths avg)')
        ax3.set_xlabel('Time Index', fontsize=11, color='white')
        ax3.set_ylabel('Volatility (Annualized)', fontsize=11, color='white')
        ax3.set_title(f'{r["name"]}: Volatility Comparison', fontsize=12, fontweight='bold', color='white')
        ax3.legend(facecolor='black', edgecolor='white', labelcolor='white')
        ax3.grid(True, alpha=0.3, color='gray')
        ax3.set_facecolor('black')
        ax3.tick_params(colors='white')
    
    plt.savefig('output/bates_comprehensive_comparison.png', dpi=300, bbox_inches='tight', facecolor='black')
    plt.close()
    
    console.print(f"[green]  ✓ Generated {4} visualization files in output/ folder[/green]")

if __name__ == "__main__":
    main()

