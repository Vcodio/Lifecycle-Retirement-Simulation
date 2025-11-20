# ====================================================================
# VERSION 5 - Bates Model MLE for Multi-Regime Returns
# --------------------------------------------------------------------
# This script performs Maxmimum liklihood estimation to fit the
# Bates stochastic volatility model (Heston + Jumps) separately across
# defined financial market regimes. This is preferable to QMLE as the 
# parameter estimates will better fit the historical skew and kurtosis
# of each regime, although it will be substantially more computationally
# intense. Both QMLE and MLE parameter estimators are available.
#
# Key Features:
# - Regime-specific fitting of Bates (Heston + Poisson Jumps) model.
# - Implementation of true MLE via Characteristic Function/Fourier Inversion.
# - Robust optimization with randomized restarts and numerical stability checks.
# - Comprehensive diagnostics: Model parameters, log-likelihood, and fitted moments.
# - Visualization of empirical vs. approximated Bates PDF for each regime.
#
# ====================================================================
# Author: VCODIO
# ====================================================================

import os
import sys
import subprocess
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.integrate import quad
from rich.console import Console
from rich.table import Table
from rich.progress import track

# ----------------------------------------------------------------------
# 1. GLOBAL CONSTANTS & INDEX MAPPING
# ----------------------------------------------------------------------

# Parameter index positions (Must be consistent across Python and Cython)
MU_IDX = 0; KAPPA_IDX = 1; THETA_IDX = 2; NU_IDX = 3; RHO_IDX = 4; V0_IDX = 5
LAM_IDX = 6; MU_J_IDX = 7; SIGMA_J_IDX = 8

IDX = {
    "mu": MU_IDX, "kappa": KAPPA_IDX, "theta": THETA_IDX, "nu": NU_IDX, 
    "rho": RHO_IDX, "v0": V0_IDX, "lam": LAM_IDX, "mu_J": MU_J_IDX, 
    "sigma_J": SIGMA_J_IDX
}

# General constants
PERIODS_PER_YEAR = 12 
DT = 1.0 / PERIODS_PER_YEAR
DENSITY_FLOOR = 1e-12
CF_INTEGRATION_LIMIT = 50.0

# The name of the Cython module
CYTHON_MODULE_NAME = 'bates_mle_cython'
console = Console(width=200)

# ----------------------------------------------------------------------
# 2. CYTHON COMPILATION LOGIC
# ----------------------------------------------------------------------

def compile_cython_module():
    """
    Creates temporary .pyx and setup.py files, compiles the Cython module.
    """
    console.print("[bold yellow]Attempting to compile Cython module for speed...[/bold yellow]")

    # Fixed Cython code with proper exception handling
    cython_code = f"""
# distutils: extra_compile_args=-O3 -ffast-math -march=native
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True

cimport cython
import numpy as np
cimport numpy as np
from scipy.integrate import quad
from libc.math cimport exp, log, sqrt, M_PI, fmax

# Manually import C complex functions from complex.h and declare them as nogil
cdef extern from "<complex.h>":
    double complex cexp(double complex z) nogil
    double complex csqrt(double complex z) nogil
    double complex clog(double complex z) nogil

# Global constants (C-level)
cdef double DENSITY_FLOOR = {DENSITY_FLOOR}
cdef double CF_INTEGRATION_LIMIT = {CF_INTEGRATION_LIMIT}
cdef double DT = {DT}

# IDX mapping (hardcoded positions for speed using Python global constants)
cdef int MU = {MU_IDX}, KAPPA = {KAPPA_IDX}, THETA = {THETA_IDX}, NU = {NU_IDX}, RHO = {RHO_IDX}, V0 = {V0_IDX}, LAM = {LAM_IDX}, MU_J = {MU_J_IDX}, SIGMA_J = {SIGMA_J_IDX}

# Fixed: Use noexcept for nogil functions (no exception propagation)
cdef double complex bates_cf_c(double u, double v_prev, double t, double[:] params) noexcept nogil:
    '''Bates Characteristic Function (CF) for the log-return (x_t) conditional on v_prev.'''
    
    # Extract parameters using indices
    cdef double mu = params[MU]
    cdef double kappa = params[KAPPA]
    cdef double theta = params[THETA]
    cdef double nu = params[NU]
    cdef double rho = params[RHO]
    cdef double lam = params[LAM]
    cdef double mu_J = params[MU_J]
    cdef double sigma_J = params[SIGMA_J]

    # Heston Components
    cdef double complex I = 0.0 + 1.0j
    cdef double complex alpha = u * u + I * u
    cdef double complex beta = kappa - I * rho * nu * u
    cdef double complex gamma = csqrt(beta * beta + alpha * nu * nu)
    
    # The complex functions C(t,u) and D(t,u)
    cdef double complex g = (beta - gamma) / (beta + gamma)
    
    cdef double complex exp_gamma_t = cexp(-gamma * t)
    cdef double complex C_t_u_numerator = (beta - gamma + (beta + gamma) * g * exp_gamma_t)
    cdef double complex C_t_u_denominator = (nu * nu * (1.0 - g * exp_gamma_t))
    cdef double complex C_t_u = (1.0 - exp_gamma_t) / C_t_u_denominator * C_t_u_numerator
            
    cdef double complex D_t_u_log_term = clog((1.0 - g * exp_gamma_t) / (1.0 - g))
    cdef double complex D_t_u = (kappa * theta / (nu * nu)) * ((beta - gamma) * t - 2.0 * D_t_u_log_term)
    
    # Jump Component CF (Log-Normal Jumps)
    cdef double complex jump_exponent = I * u * mu_J - 0.5 * u * u * sigma_J * sigma_J
    cdef double complex jump_term = lam * t * (cexp(jump_exponent) - 1.0)
    
    # Bates CF (Heston * Jump)
    cdef double complex cf = cexp(I * u * mu * t + C_t_u * v_prev + D_t_u + jump_term)
    
    return cf

# Wrapper function for the Lewis Fourier inversion integrand, called by scipy.quad
def cf_pdf_inversion_integrand_py(double u, double r_t, double v_prev, double t, double[:] params_tuple):
    '''Integrand for the Lewis Fourier inversion formula, exposed to Python/scipy.quad.'''
    
    cdef double complex cf_val
    with nogil:
        # Use the memoryview directly
        cf_val = bates_cf_c(u, v_prev, t, params_tuple)
    
    # The integrand is Re[exp(-i*u*r_t) * Phi(u)]
    cdef double complex I = 0.0 + 1.0j
    cdef double complex integrand = cexp(-I * u * r_t) * cf_val
    
    # We only need the real part for the integral
    return integrand.real

@cython.cdivision(True)
def neg_log_likelihood_c(double[:] params, double[:] returns):
    '''
    Calculates the Negative Log-Likelihood (NLL) for the Bates model.
    This is the core function called by the optimizer.
    '''
    cdef double kappa = params[KAPPA]
    cdef double theta = params[THETA]
    cdef double nu = params[NU]
    cdef double v0 = params[V0]
    cdef double lam = params[LAM]
    cdef double mu = params[MU]
    cdef double rho = params[RHO]
    cdef double mu_J = params[MU_J]
    cdef double sigma_J = params[SIGMA_J]

    cdef double r

    # Simple check for positive values where needed
    if kappa <= 0 or theta <= 0 or nu <= 0 or v0 <= 0 or lam < 0:
        return 1e12

    # Feller condition soft penalty
    cdef double feller_gap = 2.0 * kappa * theta - nu * nu
    cdef double penalty = 0.0
    if feller_gap <= 0:
        penalty = 1e6 * (1.0 + (-feller_gap))

    cdef double neg_ll = 0.0
    cdef double v_prev = v0
    
    # Pre-calculate variance propagation factors
    cdef double exp_factor = exp(-kappa * DT)
    cdef double Ev_t

    cdef double pdf_step
    cdef object result_quad
    
    try:
        for r in returns:
            result_quad = quad(
                cf_pdf_inversion_integrand_py,
                0, CF_INTEGRATION_LIMIT,
                args=(r, v_prev, DT, params)
            )
            
            pdf_step = result_quad[0] / M_PI
            
            # Bound the PDF to ensure log is finite
            pdf_step = fmax(pdf_step, DENSITY_FLOOR)
            neg_ll -= log(pdf_step)
            
            # Variance propagation
            Ev_t = theta + (v_prev - theta) * exp_factor
            v_prev = Ev_t 
            
    except Exception as e:
        return 1e12

    return neg_ll + penalty
"""

    setup_code = f"""
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
        name="{CYTHON_MODULE_NAME}",
        sources=["{CYTHON_MODULE_NAME}.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=['-O3', '-ffast-math', '-march=native'],
        language="c"
    )
]

setup(
    name='{CYTHON_MODULE_NAME}',
    ext_modules=cythonize(ext_modules, compiler_directives={{
        'cdivision': True, 'boundscheck': False, 'wraparound': False
    }}),
    zip_safe=False,
)
"""

    with tempfile.TemporaryDirectory() as tmp_dir:
        pyx_filepath = os.path.join(tmp_dir, f"{CYTHON_MODULE_NAME}.pyx")
        setup_filepath = os.path.join(tmp_dir, "setup.py")

        with open(pyx_filepath, 'w') as f:
            f.write(cython_code)
        
        with open(setup_filepath, 'w') as f:
            f.write(setup_code)

        console.print(f"[bold blue]DEBUG:[/bold blue] Writing Cython source to: [i]{pyx_filepath}[/i]")
        console.print(f"[bold blue]DEBUG:[/bold blue] Writing setup script to: [i]{setup_filepath}[/i]")

        try:
            if 'google.colab' in sys.modules:
                console.print("[bold blue]DEBUG:[/bold blue] Installing Cython...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "cython"], 
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            result = subprocess.run(
                [sys.executable, "setup.py", "build_ext", "--inplace"],
                cwd=tmp_dir,
                check=True,
                capture_output=True,
                text=True
            )
            console.print("[green]Cython compilation successful.[/green]")
            
            sys.path.append(tmp_dir)
            import importlib.util
            
            compiled_file = None
            for root, _, files in os.walk(os.path.join(tmp_dir, 'build')):
                for f in files:
                    if f.startswith(CYTHON_MODULE_NAME) and (f.endswith('.so') or f.endswith('.pyd')):
                        compiled_file = os.path.join(root, f)
                        break
                if compiled_file:
                    break
            
            if compiled_file:
                console.print(f"[bold blue]DEBUG:[/bold blue] Found compiled file: [i]{compiled_file}[/i]")
                spec = importlib.util.spec_from_file_location(CYTHON_MODULE_NAME, compiled_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                console.print(f"[bold green]Successfully imported optimized module: {CYTHON_MODULE_NAME}[/bold green]")
                return module
            else:
                raise ImportError("Could not locate compiled module file.")

        except subprocess.CalledProcessError as e:
            console.print(f"[bold red]Cython Compilation Failed:[/bold red]")
            console.print(f"Stderr:\n{e.stderr}")
            raise
        except ImportError as e:
            console.print(f"[bold red]Import Failed:[/bold red] {e}")
            raise
        
# Initialize the Cython module
try:
    optimized_bates = compile_cython_module()
    NEG_LOG_LIKELIHOOD = optimized_bates.neg_log_likelihood_c
except Exception as e:
    print(f"WARNING: Cython compilation or import failed due to: {e}. Falling back to pure Python functions.")
    optimized_bates = None
    NEG_LOG_LIKELIHOOD = None 

# ----------------------------------------------------------------------
# 3. PURE PYTHON FALLBACKS
# ----------------------------------------------------------------------

def bates_cf(u, v_prev, t, params):
    """Pure Python version of Bates Characteristic Function."""
    import cmath
    mu = params[IDX['mu']]; kappa = params[IDX['kappa']]; theta = params[IDX['theta']]
    nu = params[IDX['nu']]; rho = params[IDX['rho']]; lam = params[IDX['lam']]
    mu_J = params[IDX['mu_J']]; sigma_J = params[IDX['sigma_J']]

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
    """Pure Python version of Integrand for the Lewis Fourier inversion formula."""
    import cmath 
    cf_val = bates_cf(u, v_prev, t, params)
    integrand = cmath.exp(-1j * u * r_t) * cf_val
    return integrand.real

def cf_pdf_inversion(r_t, v_prev, params):
    """Pure Python version of PDF inversion using Lewis's integral."""
    import math
    integral, err = quad(
        cf_pdf_inversion_integrand,
        0, CF_INTEGRATION_LIMIT,
        args=(r_t, v_prev, DT, params)
    )
    pdf = integral / math.pi
    return max(pdf, DENSITY_FLOOR)

def neg_log_likelihood_py(params, returns):
    """Pure Python version of NLL."""
    import math
    if np.any(~np.isfinite(params)): return 1e12
    kappa = params[IDX['kappa']]; theta = params[IDX['theta']]
    nu = params[IDX['nu']]; v0 = params[IDX['v0']]; lam = params[IDX['lam']]
    
    if kappa <= 0 or theta <= 0 or nu <= 0 or v0 <= 0 or lam < 0: 
        return 1e12

    feller_gap = 2.0 * kappa * theta - nu ** 2
    penalty = 1e6 * (1.0 + (-feller_gap)) if feller_gap <= 0 else 0.0

    ll = 0.0
    v_prev = params[IDX['v0']]
    
    try:
        for r in returns:
            pdf_step = cf_pdf_inversion(r, v_prev, params)
            ll += math.log(pdf_step)
            
            exp_factor = math.exp(-params[IDX['kappa']] * DT)
            Ev_t = params[IDX['theta']] + (v_prev - params[IDX['theta']]) * exp_factor
            v_prev = Ev_t 
            
    except Exception:
        return 1e12

    return -ll + penalty

# Set the final NLL function based on compilation success
if NEG_LOG_LIKELIHOOD is None:
    import math, cmath
    NEG_LOG_LIKELIHOOD = neg_log_likelihood_py
    console.print("[bold red]Using pure Python fallback - Cython compilation failed.[/bold red]")


# ----------------------------------------------------------------------
# 4. APPLICATION LOGIC
# ----------------------------------------------------------------------

# --- USER CONFIGURATION ---
CSV_FILE = "regime_classification_nominal_returns.csv"
REGIME_COLUMN = "Inferred Regime ID (0=Low, 2=High)"
RETURNS_COLUMN = "Total Nominal Return (%)"
REGIMES_TO_FIT = [0.0, 1.0, 2.0]
RETURNS_ARE_PERCENT = True

# Fitting controls
NUM_RESTARTS = 6
MAXITER = 500000
USE_BOUNDS = True  # Fixed: was USE_bounds
VERBOSE = True
PLOT_K_MAX = 40

# Bounds for (mu, kappa, theta, nu, rho, v0, lam, mu_J, sigma_J)
DEFAULT_BOUNDS = [
    (-0.5, 0.5),         # mu (log-return drift)
    (0.1, 50.0),         # kappa
    (0.005, 1),          # theta (long-run variance)
    (0.5, 10.0),         # nu (vol-of-vol)
    (-1.0, 0.0),         # rho
    (0.005, 1.0),        # v0 (initial variance)
    (0.05, 20.0),        # lambda (jump intensity)
    (-5.0, 0.0),         # mu_J (jump mean)
    (0.2, 1.0),          # sigma_J (jump vol)
]


def compute_loglik_stats(params, returns):
    """Computes the log-likelihood sum and mean."""
    import math 
    
    ll = 0.0
    v_prev = params[IDX['v0']]
    
    try:
        for r in returns:
            pdf_step = cf_pdf_inversion(r, v_prev, params)
            ll += math.log(pdf_step)
            
            exp_factor = math.exp(-params[IDX['kappa']] * DT)
            Ev_t = params[IDX['theta']] + (v_prev - params[IDX['theta']]) * exp_factor
            v_prev = Ev_t 
            
    except Exception:
        return 0.0, 0.0
        
    return ll / len(returns), ll


def approx_bates_pdf(x, mu, theta, lam, mu_J, sigma_J, k_max=PLOT_K_MAX):
    """Approximate Bates PDF for plotting purposes."""
    import math 
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


def fit_bates_to_returns(returns, name="regime", restarts=NUM_RESTARTS, maxiter=MAXITER,
                         bounds_vec=DEFAULT_BOUNDS, use_bounds=USE_BOUNDS):  # Fixed: USE_BOUNDS
    """Fitting wrapper per regime (MLE version)."""
    import math 
    n = len(returns)
    if n < 8:
        console.print(f"[yellow]Too few observations for {name} ({n}); skipping.[/yellow]")
        return None

    # Empirical per-period moments -> annualize for initial guesses
    mean_period = np.mean(returns)
    var_period = np.var(returns, ddof=0)
    mu_annual = mean_period / DT
    var_annual = var_period / DT

    # Base initial vector
    init_base = np.zeros(9, dtype=float)
    init_base[IDX['mu']] = np.clip(mu_annual, bounds_vec[IDX['mu']][0], bounds_vec[IDX['mu']][1])
    init_base[IDX['kappa']] = 1.0
    init_base[IDX['theta']] = np.clip(var_annual * 0.8, bounds_vec[IDX['theta']][0], bounds_vec[IDX['theta']][1])
    init_base[IDX['nu']] = 0.5
    init_base[IDX['rho']] = -0.3
    init_base[IDX['v0']] = np.clip(var_annual * 1.2, bounds_vec[IDX['v0']][0], bounds_vec[IDX['v0']][1])
    init_base[IDX['lam']] = 0.5
    init_base[IDX['mu_J']] = -0.01
    init_base[IDX['sigma_J']] = 0.05

    # Bounds handling
    if use_bounds:
        bounds = list(bounds_vec)
    else:
        bounds = [(None, None)] * len(init_base)

    best_val = np.inf
    best_res = None
    rng = np.random.default_rng(12345 + abs(hash(name)) % 9999)

    progress_description = f"Optimizing {name} (MLE, {restarts} restarts)"
    
    # Cast returns to contiguous numpy array for Cython memoryview
    returns_arr = np.ascontiguousarray(returns, dtype=np.float64)

    for attempt in track(range(restarts), description=progress_description, console=console):
        
        # Add perturbation
        scales = np.array([0.005, 0.2, 0.1 * init_base[IDX['theta']], 0.1, 0.05,
                           0.1 * init_base[IDX['v0']], 0.1, 0.002, 0.01])
        perturb = rng.normal(scale=scales)
        x0 = init_base + perturb

        # Ensure initial guess is strictly inside the boundary
        if use_bounds:
            for j in range(len(x0)):
                lo, hi = bounds[j]
                if lo is not None and x0[j] <= lo:
                    x0[j] = lo + 1e-12
                if hi is not None and x0[j] >= hi:
                    x0[j] = hi - 1e-12

        res = minimize(NEG_LOG_LIKELIHOOD, x0,
                       args=(returns_arr,),
                       method="L-BFGS-B",
                       bounds=bounds if use_bounds else None,
                       options={"maxiter": maxiter, "ftol": 1e-8, "disp": False})

        if res.success and res.fun < best_val:
            best_val = res.fun
            best_res = res

    if best_res is None:
        console.print(f"[red]Failed to converge for {name} after {restarts} restarts.[/red]")
        return None

    # Post-fit processing
    params = best_res.x
    mu = params[IDX['mu']]; kappa = params[IDX['kappa']]; theta = params[IDX['theta']]
    nu = params[IDX['nu']]; rho = params[IDX['rho']]; v0 = params[IDX['v0']]
    lam = params[IDX['lam']]; muJ = params[IDX['mu_J']]; sigmaJ = params[IDX['sigma_J']]

    # Compute final statistics
    mean_ll, total_ll = compute_loglik_stats(params, returns_arr)

    # Approximate model mean & std per period
    mean_fit_period = mu * DT + lam * DT * muJ
    var_fit_period = theta * DT + lam * DT * (sigmaJ ** 2 + muJ ** 2)
    std_fit_period = math.sqrt(max(var_fit_period, 0.0))

    # Check bounds hit
    bounds_hit = False
    if use_bounds:
         for i, val in enumerate(params):
            lo, hi = bounds_vec[i]
            if lo is not None and abs(val - lo) < 1e-8: bounds_hit = True; break
            if hi is not None and abs(val - hi) < 1e-8: bounds_hit = True; break

    out = {
        "name": name, "N_obs": n, "mu_annual": mu, "kappa": kappa, 
        "theta_annual": theta, "nu": nu, "rho": rho, "v0": v0,
        "lambda_per_year": lam, "mu_J": muJ, "sigma_J": sigmaJ,
        "mean_ll_per_obs": mean_ll, "total_negloglik": -total_ll,
        "model_mean_per_period": mean_fit_period, "model_std_per_period": std_fit_period,
        "bounds_hit": bounds_hit,
    }
    return out


def plot_empirical_vs_bates(returns, params, regime_id, save_fig=False):
    """Plotting function."""
    mu = params[IDX['mu']]; theta = params[IDX['theta']]; lam = params[IDX['lam']]
    muJ = params[IDX['mu_J']]; sigmaJ = params[IDX['sigma_J']]

    xs = np.linspace(min(returns), max(returns), 400)
    ys = approx_bates_pdf(xs, mu, theta, lam, muJ, sigmaJ, k_max=PLOT_K_MAX) 

    plt.figure(figsize=(8,5))
    plt.hist(returns, bins=30, density=True, alpha=0.6, color='gray', label='Empirical (Log-Returns)')
    plt.plot(xs, ys, 'r-', lw=2, label='Bates (QMLE Approx) PDF')
    plt.title(f"Regime {regime_id}: Empirical vs Bates (MLE Fit) PDF")
    plt.xlabel("Return (Log-Return per period)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    if save_fig:
        safe = str(regime_id).replace(" ", "_")
        plt.savefig(f"bates_mle_regime_{safe}.png", dpi=150)
    plt.show()


def print_parameter_table(rows):
    """Print results table 1."""
    if not rows: console.print("[yellow]No parameter rows to show[/yellow]"); return
    table = Table(title="Bates MLE Fit Parameters", header_style="bold cyan", show_lines=True)
    table.add_column("name", style="bold")
    cols = ["N_obs", "mu_annual", "kappa", "theta_annual", "nu", "rho", "v0", "lambda_per_year", "mu_J", "sigma_J"]
    for c in cols: table.add_column(c, justify="right", overflow="fold")
    for r in rows:
        table.add_row(r["name"], str(r["N_obs"]), f"{r['mu_annual']:.6f}", f"{r['kappa']:.6f}", 
                      f"{r['theta_annual']:.6f}", f"{r['nu']:.6f}", f"{r['rho']:.6f}", 
                      f"{r['v0']:.6f}", f"{r['lambda_per_year']:.6f}", f"{r['mu_J']:.6f}", f"{r['sigma_J']:.6f}")
    console.print(table)

def print_diagnostics_table(rows):
    """Print results table 2."""
    if not rows: console.print("[yellow]No diagnostics rows to show[/yellow]"); return
    table = Table(title="Fit Diagnostics & Moments", header_style="bold magenta", show_lines=True)
    table.add_column("name", style="bold")
    diag_cols = ["mean_ll_per_obs", "total_negloglik", "model_mean_per_period", "model_std_per_period", "bounds_hit"]
    for c in diag_cols: table.add_column(c, justify="right", overflow="fold")
    for r in rows:
        table.add_row(r["name"], f"{r['mean_ll_per_obs']:.6f}", f"{-r['total_negloglik']:.6f}", 
                      f"{r['model_mean_per_period']:.6e}", f"{r['model_std_per_period']:.6e}", str(r["bounds_hit"]))
    console.print(table)


# -------------------------
# MAIN EXECUTION
# -------------------------
def main():
    if not os.path.exists(CSV_FILE):
        console.print(f"[red]CSV file not found: {CSV_FILE}[/red]")
        # Create a dummy file for execution environment testing
        console.print("[yellow]Creating dummy CSV file for demonstration.[/yellow]")
        dummy_data = {
            REGIME_COLUMN: [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 0.0, 1.0, 2.0] * 10,
            RETURNS_COLUMN: np.random.normal(loc=0.5, scale=1.5, size=120) 
        }
        pd.DataFrame(dummy_data).to_csv(CSV_FILE, index=False)
        console.print(f"[green]Dummy data created in {CSV_FILE}[/green]")

    df = pd.read_csv(CSV_FILE)
    if REGIME_COLUMN not in df.columns or RETURNS_COLUMN not in df.columns:
        console.print("[red]CSV missing required columns. Check REGIME_COLUMN and RETURNS_COLUMN settings.[/red]")
        return

    results = []
    for r in REGIMES_TO_FIT:
        mask = df[REGIME_COLUMN] == r
        if mask.sum() == 0: mask = df[REGIME_COLUMN].astype(str) == str(r)
        regime_vals = df.loc[mask, RETURNS_COLUMN].values.astype(float)
        if len(regime_vals) == 0:
            console.print(f"[yellow]No data for regime {r}; skipping.[/yellow]"); continue

        simple_returns = regime_vals / 100.0 if RETURNS_ARE_PERCENT else regime_vals.copy()
        
        # Guard against zero or negative returns before taking log
        if np.any(simple_returns <= -1.0):
             console.print(f"[red]Regime {r}: Contains returns <= -100%. Cannot compute log-returns. Skipping.[/red]"); continue
             
        returns = np.log(1.0 + simple_returns)
        
        console.print(f"[green]Fitting regime {r} with {len(returns)} observations using Full MLE[/green]")
        
        fit = fit_bates_to_returns(returns, name=f"regime_{r}", restarts=NUM_RESTARTS, maxiter=MAXITER,
                                   bounds_vec=DEFAULT_BOUNDS, use_bounds=USE_BOUNDS)
        if fit:
            results.append(fit)
            params_vec = np.zeros(9)
            params_vec[IDX['mu']] = fit['mu_annual']; params_vec[IDX['theta']] = fit['theta_annual']
            params_vec[IDX['lam']] = fit['lambda_per_year']; params_vec[IDX['mu_J']] = fit['mu_J']
            params_vec[IDX['sigma_J']] = fit['sigma_J']
            plot_empirical_vs_bates(returns, params_vec, r, save_fig=False)

    if results:
        print_parameter_table(results)
        print_diagnostics_table(results)
        pd.DataFrame(results).to_csv("Bates_per_regime_MLE_results.csv", index=False)
        console.print("[green]Saved Bates_per_regime_MLE_results.csv[/green]")
    else:
        console.print("[yellow]No successful fits.[/yellow]")

if __name__ == "__main__":
    main()
