"""
# =============================================================
# VERSION 6.3 — Lifecycle Retirement Simulation - Fixed Withdrawal
# ================================================================
# This script simulates a complete financial lifecycle, from the
# accumulation phase (working and saving) to the retirement phase
# (withdrawing from the portfolio). It uses a stochastic model to
# account for market returns, inflation, salary growth, and even
# unemployment. This uses a Fixed Spending Rule, withdrawing
# spending_real from the portfolio every year.
#
# Key Features:
# - Stage 1: Calculates the required principal to retire at various ages
#            within a specified success rate.
# - Stage 2: Runs a large number of simulations to determine the
#            distribution of possible retirement ages.
# - Stochastic Modeling: Uses a Bates jump-diffusion model for market returns
#            (parametric) or block bootstrap from historical data, with normal
#            distributions for inflation, salary, and savings.
# - Cython Acceleration: Uses compiled Cython extensions for 10-50x speedup
#            on computational hotspots (automatic fallback to Python if unavailable).
# - Block Bootstrap Option: Can use historical data blocks instead of parametric
#            model to preserve historical correlations between returns and inflation.
# - CSV Export: Exports summary tables and detailed simulation paths for
#               analysis and debugging.
#
# Changes from version 5.5:
# - Added Cython acceleration for ~10-50x performance improvement
# - Added block bootstrap option to use historical data instead of parametric model
# - Improved multiprocessing support for parallel simulations
# - Enhanced inflation modeling with synchronized returns/inflation from bootstrap
# - Better error handling and fallback mechanisms
# - Google Colab compatible with inline Cython compilation
#
# Changes from version 4:
# - Upgraded from Stochastic Jump Diffusion model to a Bates Model
# - Upgraded from Annual steps to monthly steps
# - Unemployment probability now coupled with exit probability so unemployment
#   doesn't just last one step
# - Several bug fixes and performance enhancements
#
# Dependencies:
#   pip install numpy pandas tqdm rich matplotlib cython
#
# For Google Colab:
#   1. Run: !pip install cython numpy pandas tqdm rich matplotlib
#   2. Then run this script (Cython will compile inline)
#
# ------------------------------------------------
# Author: VCODIO
# ------------------------------------------------
"""

# ============================================================================
# STEP 1: INSTALL AND LOAD CYTHON
# ============================================================================
import sys
import os

# Add parent directories to path to find compiled Cython module
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
_grandparent_dir = os.path.dirname(_parent_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
if _grandparent_dir not in sys.path:
    sys.path.insert(0, _grandparent_dir)

# Add cython folder to path
_cython_dir = os.path.join(_grandparent_dir, 'cython')
if os.path.exists(_cython_dir) and _cython_dir not in sys.path:
    sys.path.insert(0, _cython_dir)

# Try to import the compiled Cython module (don't print here - printed in main() only)
CYTHON_AVAILABLE = False
try:
    # First, try importing the compiled module (for standalone Python)
    from lrs_cython import (
        simulate_monthly_return_svj_cython,
        calculate_max_drawdown_cython,
        update_unemployment_cython
    )
    CYTHON_AVAILABLE = True
except ImportError:
    # Fallback: Try Jupyter/IPython magic (for Colab/Jupyter)
    try:
        get_ipython().run_line_magic('load_ext', 'cython')
        # If we get here, we're in Jupyter - compile inline
        get_ipython().run_cell_magic('cython', '', '''
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

import numpy as np
cimport numpy as cnp
from libc.math cimport exp, sqrt, log
cimport cython

ctypedef cnp.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef tuple simulate_monthly_return_svj_cython(
    object rng_local,
    dict params_annual,
    double current_variance
):
    cdef double dt = 1.0 / 12.0
    cdef double kappa = params_annual["kappa"]
    cdef double theta = params_annual["theta"]
    cdef double nu = params_annual["nu"]
    cdef double rho = params_annual["rho"]
    cdef double jump_intensity = params_annual["lam"]
    cdef double jump_mean = params_annual["mu_J"]
    cdef double jump_std_dev = params_annual["sigma_J"]
    cdef double mu_annual = params_annual["mu"]
    cdef double z1, z2, z_v, z_s
    cdef double v, dv, v_new
    cdef int num_jumps
    cdef double jump_component = 0.0
    cdef double jump_drift_correction
    cdef double drift_component, diffusion_component, total_log_return
    cdef double simple_return
    cdef double rho_complement
    z1 = rng_local.normal(0.0, 1.0)
    z2 = rng_local.normal(0.0, 1.0)
    z_v = z1
    rho_complement = 1.0 - rho * rho
    if rho_complement < 0.0:
        rho_complement = 0.0
    z_s = rho * z1 + sqrt(rho_complement) * z2
    v = current_variance
    if v < 0.0:
        v = 0.0
    dv = kappa * (theta - v) * dt + nu * sqrt(v) * sqrt(dt) * z_v
    v_new = v + dv
    if v_new < 1e-8:
        v_new = 1e-8
    num_jumps = rng_local.poisson(jump_intensity * dt)
    if num_jumps > 0:
        jump_component = rng_local.normal(jump_mean, jump_std_dev, size=num_jumps).sum()
    jump_drift_correction = jump_intensity * (exp(jump_mean + 0.5 * jump_std_dev * jump_std_dev) - 1.0)
    drift_component = (mu_annual - jump_drift_correction - 0.5 * v) * dt
    diffusion_component = sqrt(v) * sqrt(dt) * z_s
    total_log_return = drift_component + diffusion_component + jump_component
    simple_return = exp(total_log_return) - 1.0
    return (simple_return, v_new)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double calculate_max_drawdown_cython(cnp.ndarray[DTYPE_t, ndim=1] series):
    cdef int n = series.shape[0]
    if n == 0:
        return 0.0
    cdef double peak = series[0]
    cdef double max_dd = 0.0
    cdef double current_dd
    cdef int i
    for i in range(n):
        if series[i] > peak:
            peak = series[i]
        current_dd = (series[i] - peak) / peak
        if current_dd < max_dd:
            max_dd = current_dd
    return max_dd

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef bint update_unemployment_cython(
    list return_buffer,
    double monthly_return,
    bint is_unemployed,
    double baseline_exit,
    double entry_prob_annual,
    double rho,
    object rng_local
):
    cdef double lagged_return, exit_prob, entry_prob
    cdef double rand_val
    return_buffer.append(monthly_return)
    if len(return_buffer) > 12:
        return_buffer.pop(0)
    lagged_return = sum(return_buffer)
    exit_prob = baseline_exit * (1.0 - rho * lagged_return)
    if exit_prob < 0.01:
        exit_prob = 0.01
    elif exit_prob > 0.9:
        exit_prob = 0.9
    if not is_unemployed:
        entry_prob = entry_prob_annual / 12.0
        rand_val = rng_local.random()
        if rand_val < entry_prob:
            return True
    else:
        rand_val = rng_local.random()
        if rand_val < exit_prob:
            return False
    return is_unemployed
''')
        CYTHON_AVAILABLE = True
    except (NameError, AttributeError):
        # Not in Jupyter and compiled module not available
        CYTHON_AVAILABLE = False

# ============================================================================
# STEP 3: STANDARD PYTHON IMPORTS
# ============================================================================

import numpy as np
import pandas as pd
import logging
import warnings
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

try:
    from rich.console import Console
    from rich.table import Table
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False

try:
    import matplotlib.pyplot as plt
    plt.style.use('dark_background')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ============================================================================
# CONFIGURATION CLASSES
# ============================================================================

class SimulationConfig:
    def __init__(self):
        self.initial_age = 25
        self.death_age = 100
        self.initial_portfolio = 180_000
        self.annual_income_real = 77_000.0
        self.annual_salary_growth_rate = 0.07
        self.salary_growth_rate_std_dev = 0.03
        self.savings_rate = 0.25
        self.savings_rate_std_dev = 0.05
        self.unemployment_prob = 0.1
        self.unemployment_income_multiplier = 0.25
        self.unemployment_savings_rate = -0.1
        self.spending_real = 70_000.0
        self.mean_inflation_geometric = 0.025
        self.std_inflation = 0.025
        self.social_security_real = 25_000.0
        self.social_security_start_age = 67
        self.include_social_security = True
        self.num_outer = 10000
        self.num_nested = 5000
        self.success_target = 0.95
        self.generate_csv_summary = False
        self.num_sims_to_export = 50
        self.seed = None
        self.num_workers = max(1, mp.cpu_count() - 1)
        self.output_directory = 'output'  # Directory to save plots and CSV files
        self.use_principal_deviation_threshold = True  # Toggle for deviation threshold
        self.principal_deviation_threshold = 0.07  # 10% deviation threshold (as fraction)
        
        # Block bootstrap configuration
        self.use_block_bootstrap = True  # Toggle to use block bootstrap instead of parametric model
        self.bootstrap_csv_path = 'data/VCEA - Block Bootstrap.csv'  # Path to CSV file
        self.portfolio_column_name = "Vcodio's Excellent Adventure"  # Column name for portfolio returns (generalized)
        self.inflation_column_name = 'Inflation'  # Column name for inflation data
        self.block_length_years = 10  # Block length in years (default 10 years = 120 months)
        self.block_overlapping = True  # Toggle for overlapping vs non-overlapping blocks
        # Note: Blocks switch every block_length_years. Each new block is randomly sampled from history.
        # For example, with block_length_years=10, an 80-year simulation uses 8 different random 10-year blocks.

        self.params = {
            "mu": 0.0928,
            "kappa": 1.189,
            "theta": 0.0201,
            "nu": 0.0219,
            "rho": -0.714,
            "v0": 0.0201,
            "lam": 0.353,
            "mu_J": -0.007,
            "sigma_J": 0.0328,
        }

    def validate(self):
        errors = []
        if not (0 < self.initial_age < self.death_age):
            errors.append("Initial age must be between 0 and death age")
        if self.initial_portfolio <= 0:
            errors.append("Initial portfolio must be positive")
        if errors:
            raise ValueError("Parameter validation failed:\n" + "\n".join(errors))
        logger.info("All parameters validated successfully")


class UnemploymentConfig:
    def __init__(self, baseline_exit=0.25, entry_prob_annual=0.10,
                 lag_months=12, rho=-0.25):
        self.baseline_exit = baseline_exit
        self.entry_prob_annual = entry_prob_annual
        self.lag_months = lag_months
        self.rho = rho


# ============================================================================
# WRAPPER FUNCTIONS (Use Cython if available, else Python)
# ============================================================================

def simulate_monthly_return_svj(rng_local, params_annual, current_variance):
    """Wrapper that uses Cython version if available"""
    if CYTHON_AVAILABLE:
        return simulate_monthly_return_svj_cython(rng_local, params_annual, current_variance)
    else:
        # Python fallback
        dt = 1.0 / 12.0
        kappa = params_annual["kappa"]
        theta = params_annual["theta"]
        nu = params_annual["nu"]
        rho = params_annual["rho"]
        jump_intensity = params_annual["lam"]
        jump_mean = params_annual["mu_J"]
        jump_std_dev = params_annual["sigma_J"]
        mu_annual = params_annual["mu"]

        z1 = rng_local.normal(0.0, 1.0)
        z2 = rng_local.normal(0.0, 1.0)
        z_v = z1
        z_s = rho * z1 + np.sqrt(max(0.0, 1.0 - rho**2)) * z2

        v = current_variance
        dv = kappa * (theta - v) * dt + nu * np.sqrt(max(v, 0.0)) * np.sqrt(dt) * z_v
        v_new = v + dv
        v_new = max(v_new, 1e-8)

        num_jumps = rng_local.poisson(jump_intensity * dt)
        jump_component = 0.0
        if num_jumps > 0:
            jump_component = rng_local.normal(jump_mean, jump_std_dev, size=num_jumps).sum()

        jump_drift_correction = jump_intensity * (np.exp(jump_mean + 0.5 * jump_std_dev**2) - 1.0)
        drift_component = (mu_annual - jump_drift_correction - 0.5 * v) * dt
        diffusion_component = np.sqrt(max(v, 0.0)) * np.sqrt(dt) * z_s
        total_log_return = drift_component + diffusion_component + jump_component
        simple_return = np.exp(total_log_return) - 1.0

        return simple_return, v_new


def calculate_max_drawdown(series):
    """Wrapper for max drawdown calculation"""
    if len(series) == 0:
        return 0.0

    if CYTHON_AVAILABLE and isinstance(series, np.ndarray):
        return calculate_max_drawdown_cython(series.astype(np.float64))
    else:
        # Python fallback
        series = np.array(series)
        peak_series = np.maximum.accumulate(series)
        drawdowns = (series - peak_series) / peak_series
        return np.min(drawdowns) if drawdowns.size > 0 else 0.0


class Unemployment:
    """Unemployment class that uses Cython if available"""
    def __init__(self, config, rng_local):
        self.config = config
        self.rng_local = rng_local
        self.return_buffer = [0.0] * config.lag_months
        self.is_unemployed = False

    def update(self, monthly_return):
        if CYTHON_AVAILABLE:
            self.is_unemployed = update_unemployment_cython(
                self.return_buffer,
                monthly_return,
                self.is_unemployed,
                self.config.baseline_exit,
                self.config.entry_prob_annual,
                self.config.rho,
                self.rng_local
            )
        else:
            # Python fallback
            self.return_buffer.append(monthly_return)
            if len(self.return_buffer) > self.config.lag_months:
                self.return_buffer.pop(0)

            lagged_return = np.sum(self.return_buffer)
            exit_prob = self.config.baseline_exit * (1.0 - self.config.rho * lagged_return)
            exit_prob = np.clip(exit_prob, 0.01, 0.9)

            if not self.is_unemployed:
                entry_prob = self.config.entry_prob_annual / 12.0
                if self.rng_local.random() < entry_prob:
                    self.is_unemployed = True
            else:
                if self.rng_local.random() < exit_prob:
                    self.is_unemployed = False

        return self.is_unemployed


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def convert_geometric_to_arithmetic(mean_geometric, std_dev):
    return mean_geometric + 0.5 * (std_dev**2)

def calculate_nominal_value(real_value, current_age_in_years, target_age_in_years,
                           mean_inflation):
    years = target_age_in_years - current_age_in_years
    return real_value * (1 + mean_inflation)**years

def print_rich_table(df, title):
    if HAS_RICH:
        table = Table(title=title, title_style="bold magenta",
                     header_style="bold cyan")
        for col in df.columns:
            table.add_column(str(col), justify="right")
        for _, row in df.iterrows():
            table.add_row(*[str(item) for item in row])
        console.print(table)
    else:
        print(f"\n--- {title} ---")
        print(df.to_string())

def export_to_csv(data, filename, output_dir='output'):
    try:
        df = pd.DataFrame(data)
        import os
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        logger.info(f"Data successfully exported to '{filepath}'")
    except Exception as e:
        logger.error(f"Error exporting to CSV '{filepath}': {e}")

def export_detailed_simulations_to_csv(sim_data, filename, output_dir='output'):
    if not sim_data:
        return
    try:
        flat_data = [entry for sim in sim_data for entry in sim]
        df = pd.DataFrame(flat_data)
        all_expected_cols = [
            'SIM_ID', 'AGE', 'RETIRED?', 'PORTFOLIO_VALUE', 'VOLATILITY',
            'REQUIRED_REAL_PRINCIPAL', 'WITHDRAWAL_RATE', 'REQUIRED_NOMINAL_PRINCIPAL',
            'NOMINAL_DESIRED_CONSUMPTION', 'REAL_DESIRED_CONSUMPTION',
            'ANNUAL_INFLATION', 'CUMULATIVE_INFLATION',
            'REAL_SOCIAL_SECURITY_BENEFIT', 'NOMINAL_SOCIAL_SECURITY_BENEFIT',
            'SAVINGS_RATE', 'SALARY_REAL', 'SALARY_NOMINAL', 'EMPLOYED?',
            'DOLLARS_SAVED', 'REAL_SALARY_GROWTH_RATE', 'NOMINAL_SALARY_GROWTH_RATE',
            'MONTHLY_PORTFOLIO_RETURN', 'CUMULATIVE_PORTFOLIO_RETURN'
        ]
        for col in all_expected_cols:
            if col not in df.columns:
                df[col] = None
        import os
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False, columns=all_expected_cols)
        logger.info(f"Detailed simulations exported to '{filepath}'")
    except Exception as e:
        logger.error(f"Error exporting detailed simulations: {e}")


# ============================================================================
# BLOCK BOOTSTRAP FUNCTIONS
# ============================================================================

def load_and_convert_to_monthly_returns(csv_path, portfolio_column_name, inflation_column_name):
    """
    Load CSV file and convert daily returns to monthly returns.
    
    Args:
        csv_path: Path to CSV file
        portfolio_column_name: Name of column containing portfolio values
        inflation_column_name: Name of column containing inflation values
    
    Returns:
        tuple: (monthly_returns, monthly_inflation, monthly_dates)
    """
    try:
        df = pd.read_csv(csv_path)
        
        # Validate columns exist
        if portfolio_column_name not in df.columns:
            raise ValueError(f"Column '{portfolio_column_name}' not found in CSV. Available columns: {list(df.columns)}")
        if inflation_column_name not in df.columns:
            raise ValueError(f"Column '{inflation_column_name}' not found in CSV. Available columns: {list(df.columns)}")
        
        # Convert Date column to datetime - handle mixed formats
        df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
        
        # Check for any NaT (Not a Time) values that couldn't be parsed
        if df['Date'].isna().any():
            logger.warning(f"Some dates could not be parsed. Dropping {df['Date'].isna().sum()} rows with invalid dates.")
            df = df.dropna(subset=['Date'])
        
        # Calculate daily returns from portfolio values
        # Assuming the values are cumulative (starting at 10000)
        portfolio_values = df[portfolio_column_name].values
        daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Calculate daily inflation changes
        inflation_values = df[inflation_column_name].values
        daily_inflation_changes = np.diff(inflation_values) / inflation_values[:-1]
        
        # Get dates (excluding first date since we lost it in diff)
        dates = df['Date'].values[1:]
        
        # Group by year-month and aggregate to monthly
        df_returns = pd.DataFrame({
            'Date': dates,
            'DailyReturn': daily_returns,
            'DailyInflation': daily_inflation_changes
        })
        df_returns['YearMonth'] = df_returns['Date'].dt.to_period('M')
        
        # Aggregate daily returns to monthly returns
        # For returns: compound them (1 + r1) * (1 + r2) * ... - 1
        # CRITICAL: Both DailyReturn and DailyInflation are grouped by the SAME YearMonth,
        # ensuring monthly_returns[i] and monthly_inflation[i] are from the exact same historical month.
        # This preserves the correlation between market returns and inflation.
        monthly_returns_data = df_returns.groupby('YearMonth').agg({
            'DailyReturn': lambda x: np.prod(1 + x) - 1.0,
            'DailyInflation': lambda x: np.prod(1 + x) - 1.0,
            'Date': 'first'
        }).reset_index()
        
        monthly_returns = monthly_returns_data['DailyReturn'].values
        monthly_inflation = monthly_returns_data['DailyInflation'].values
        monthly_dates = monthly_returns_data['Date'].values
        
        # Verify alignment: same length ensures index alignment
        if len(monthly_returns) != len(monthly_inflation):
            raise ValueError(f"Alignment error: monthly_returns and monthly_inflation must have same length. "
                           f"Got {len(monthly_returns)} and {len(monthly_inflation)}")
        
        logger.info(f"Loaded {len(monthly_returns)} monthly returns from CSV")
        logger.info(f"Monthly returns range: {np.min(monthly_returns):.4f} to {np.max(monthly_returns):.4f}")
        logger.info(f"Loaded {len(monthly_inflation)} monthly inflation values from CSV")
        logger.info(f"Monthly inflation range: {np.min(monthly_inflation):.4f} to {np.max(monthly_inflation):.4f}")
        logger.info(f"✓ Returns and inflation are aligned by date index - correlation preserved")
        
        return monthly_returns, monthly_inflation, monthly_dates
        
    except Exception as e:
        logger.error(f"Error loading CSV file '{csv_path}': {e}")
        raise


class BlockBootstrap:
    """
    Block bootstrap sampler for monthly returns.
    Supports both overlapping and non-overlapping blocks.
    """
    def __init__(self, monthly_returns, monthly_inflation, block_length_months, 
                 overlapping=True, rng=None):
        """
        Initialize block bootstrap.
        
        Args:
            monthly_returns: Array of monthly returns
            monthly_inflation: Array of monthly inflation values
            block_length_months: Length of each block in months (e.g., 120 for 10 years)
            overlapping: If True, use overlapping blocks; if False, use non-overlapping
            rng: Random number generator (default: new generator)
        
        IMPORTANT: monthly_returns[i] and monthly_inflation[i] MUST correspond to the same
        historical month/date to preserve the correlation between market returns and inflation.
        This is ensured by load_and_convert_to_monthly_returns() which groups both by YearMonth.
        """
        self.monthly_returns = np.array(monthly_returns)
        self.monthly_inflation = np.array(monthly_inflation)
        
        # Verify arrays are the same length (safety check)
        if len(self.monthly_returns) != len(self.monthly_inflation):
            raise ValueError(f"monthly_returns and monthly_inflation must have the same length. "
                           f"Got {len(self.monthly_returns)} and {len(self.monthly_inflation)}")
        self.block_length_months = int(block_length_months)
        self.overlapping = overlapping
        self.rng = rng if rng is not None else np.random.default_rng()
        
        # Create blocks (verbose=False to avoid repeated logging)
        self._create_blocks(verbose=False)
        
    def _create_blocks(self, verbose=False):
        """Create blocks from monthly data."""
        n_months = len(self.monthly_returns)
        
        if self.overlapping:
            # Overlapping blocks: can start at any month (1-month step between block starts)
            # Last valid start index: n_months - block_length_months
            self.num_blocks = n_months - self.block_length_months + 1
            self.block_starts = np.arange(self.num_blocks)  # Blocks start at [0, 1, 2, 3, ..., num_blocks-1]
            # This creates blocks with 1-month step size, maximizing the number of available blocks
        else:
            # Non-overlapping blocks: blocks don't overlap
            self.num_blocks = n_months // self.block_length_months
            self.block_starts = np.arange(self.num_blocks) * self.block_length_months
            # This creates blocks with block_length_months step size (e.g., 120 months for 10-year blocks)
        
        if self.num_blocks == 0:
            raise ValueError(f"Not enough data for block length {self.block_length_months}. "
                           f"Need at least {self.block_length_months} months, got {n_months}")
        
        # Only log on first creation (when verbose=True)
        if verbose:
            step_size = 1 if self.overlapping else self.block_length_months
            logger.info(f"Created {self.num_blocks} {'overlapping' if self.overlapping else 'non-overlapping'} "
                       f"blocks of length {self.block_length_months} months "
                       f"(step size: {step_size} month{'s' if step_size > 1 else ''} between block starts)")
    
    def sample_block(self):
        """
        Sample a random block from the data.
        
        Returns:
            tuple: (returns_block, inflation_block) - arrays of length block_length_months
            Note: Returns and inflation are sampled together from the same historical period
                  to maintain their correlation and synchronization.
        """
        block_idx = self.rng.integers(0, self.num_blocks)
        start_idx = self.block_starts[block_idx]
        end_idx = start_idx + self.block_length_months
        
        # CRITICAL: Use the SAME start_idx and end_idx for both arrays to ensure
        # returns_block[i] and inflation_block[i] are from the exact same historical month.
        # This preserves the historical correlation between S&P 500 returns and inflation.
        returns_block = self.monthly_returns[start_idx:end_idx].copy()
        inflation_block = self.monthly_inflation[start_idx:end_idx].copy()
        
        return returns_block, inflation_block
    
    def sample_sequence(self, num_months, verbose=False):
        """
        Sample a sequence of returns of specified length using block bootstrap.
        
        Each block is 10 years (120 months) long. A new random block is sampled
        every 10 years. For example, an 80-year simulation would use 8 different
        random 10-year blocks from historical data.
        
        Args:
            num_months: Number of months to sample
            verbose: If True, log when blocks switch
        
        Returns:
            tuple: (returns_sequence, inflation_sequence) - arrays of length num_months
            Note: Returns and inflation are sampled together from the same blocks to maintain
                  their historical correlation and ensure they stay synchronized.
        """
        returns_sequence = []
        inflation_sequence = []
        
        remaining_months = num_months
        block_number = 0
        while remaining_months > 0:
            returns_block, inflation_block = self.sample_block()
            block_number += 1
            
            if verbose:
                logger.info(f"Block {block_number}: Sampling {len(returns_block)} months "
                          f"({remaining_months} months remaining)")
            
            # Take only what we need
            take_months = min(remaining_months, len(returns_block))
            returns_sequence.extend(returns_block[:take_months])
            inflation_sequence.extend(inflation_block[:take_months])
            
            remaining_months -= take_months
        
        if verbose:
            logger.info(f"Sampled {block_number} blocks for {num_months} total months")
        
        return np.array(returns_sequence[:num_months]), np.array(inflation_sequence[:num_months])


# Global cache for bootstrap data (loaded once)
_bootstrap_data_cache = None

def load_bootstrap_data(config):
    """
    Load bootstrap data once and cache it globally.
    
    Args:
        config: SimulationConfig instance
    
    Returns:
        tuple: (monthly_returns, monthly_inflation) or None if bootstrap disabled
    """
    global _bootstrap_data_cache
    
    if not config.use_block_bootstrap:
        return None
    
    # Return cached data if available
    if _bootstrap_data_cache is not None:
        return _bootstrap_data_cache
    
    try:
        monthly_returns, monthly_inflation, _ = load_and_convert_to_monthly_returns(
            config.bootstrap_csv_path,
            config.portfolio_column_name,
            config.inflation_column_name
        )
        
        _bootstrap_data_cache = (monthly_returns, monthly_inflation)
        logger.info(f"Bootstrap data loaded and cached: {len(monthly_returns)} monthly returns")
        
        return _bootstrap_data_cache
        
    except Exception as e:
        logger.error(f"Failed to load bootstrap data: {e}")
        raise


# Track if we've logged block structure info
_block_structure_logged = False

def create_block_bootstrap_sampler(config, rng, monthly_returns=None, monthly_inflation=None):
    """
    Create a block bootstrap sampler from configuration.
    
    Args:
        config: SimulationConfig instance
        rng: Random number generator
        monthly_returns: Pre-loaded monthly returns (optional, will load if None)
        monthly_inflation: Pre-loaded monthly inflation (optional, will load if None)
    
    Returns:
        BlockBootstrap instance or None if block bootstrap is disabled
    """
    global _block_structure_logged
    
    if not config.use_block_bootstrap:
        return None
    
    try:
        # Use provided data or load it
        if monthly_returns is None or monthly_inflation is None:
            data = load_bootstrap_data(config)
            if data is None:
                return None
            monthly_returns, monthly_inflation = data
        
        block_length_months = int(config.block_length_years * 12)
        
        # Log block structure info only once
        if not _block_structure_logged:
            n_months = len(monthly_returns)
            if config.block_overlapping:
                num_blocks = n_months - block_length_months + 1
                step_size = 1  # Overlapping blocks have 1-month step between starts
            else:
                num_blocks = n_months // block_length_months
                step_size = block_length_months  # Non-overlapping blocks have block_length step
            logger.info(f"Block structure computed: {num_blocks} {'overlapping' if config.block_overlapping else 'non-overlapping'} "
                       f"blocks of length {block_length_months} months available "
                       f"(step size: {step_size} month{'s' if step_size > 1 else ''} between block starts). "
                       f"Each simulation will randomly sample different blocks from this pool of {num_blocks} blocks.")
            _block_structure_logged = True
        
        bootstrap_sampler = BlockBootstrap(
            monthly_returns,
            monthly_inflation,
            block_length_months,
            overlapping=config.block_overlapping,
            rng=rng
        )
        
        return bootstrap_sampler
        
    except Exception as e:
        logger.error(f"Failed to create block bootstrap sampler: {e}")
        raise


# ============================================================================
# SIMULATION CORE FUNCTIONS
# ============================================================================

def simulate_withdrawals(start_portfolio, start_age, rng_local, params_annual,
                         spending_annual_real, include_social_security,
                         social_security_real, social_security_start_age,
                         config, bootstrap_sampler=None):
    portfolio = float(start_portfolio)
    age_in_months = int(start_age * 12)
    portfolio_history = [portfolio]
    
    # Initialize variance for parametric model (only used if not using bootstrap)
    current_variance = params_annual["v0"] if not config.use_block_bootstrap else params_annual["v0"]

    current_annual_spending_nominal = spending_annual_real
    current_monthly_spending_nominal = current_annual_spending_nominal / 12.0
    current_social_security_nominal = social_security_real
    current_monthly_social_security_nominal = current_social_security_nominal / 12.0

    # Pre-sample all returns and inflation if using block bootstrap
    use_bootstrap = False
    if config.use_block_bootstrap and bootstrap_sampler is not None:
        try:
            total_months = int((config.death_age - start_age) * 12)
            bootstrap_returns, bootstrap_inflation = bootstrap_sampler.sample_sequence(total_months)
            bootstrap_month_idx = 0
            use_bootstrap = True
        except Exception as e:
            logger.warning(f"Block bootstrap failed, falling back to parametric model: {e}")
            bootstrap_returns = None
            bootstrap_inflation = None
            bootstrap_month_idx = None
            use_bootstrap = False
    else:
        bootstrap_returns = None
        bootstrap_inflation = None
        bootstrap_month_idx = None
        use_bootstrap = False

    while (age_in_months / 12.0) < config.death_age:
        if (age_in_months % 12) == 0 and age_in_months > int(start_age * 12):
            if use_bootstrap and bootstrap_inflation is not None:
                # Calculate annual inflation from the previous 12 months of bootstrap inflation
                # At the start of a new year, bootstrap_month_idx represents months already processed
                # So we calculate inflation from the previous 12 months (indices bootstrap_month_idx-12 to bootstrap_month_idx-1)
                # which corresponds to bootstrap_inflation[bootstrap_month_idx-12:bootstrap_month_idx]
                year_start_idx = max(0, bootstrap_month_idx - 12)
                year_end_idx = bootstrap_month_idx
                if year_end_idx > year_start_idx and year_end_idx - year_start_idx == 12:
                    monthly_inflations = bootstrap_inflation[year_start_idx:year_end_idx]
                    annual_inflation = np.prod(1.0 + monthly_inflations) - 1.0
                    # Inflation from bootstrap data is being used (synchronized with returns from same blocks)
                else:
                    # Fallback if we don't have enough data yet
                    annual_inflation = rng_local.normal(config.mean_inflation_geometric,
                                                       config.std_inflation)
            else:
                annual_inflation = rng_local.normal(config.mean_inflation_geometric,
                                                   config.std_inflation)
            annual_inflation = max(annual_inflation, -0.99)

            current_annual_spending_nominal *= (1.0 + annual_inflation)
            current_monthly_spending_nominal = current_annual_spending_nominal / 12.0
            current_social_security_nominal *= (1.0 + annual_inflation)
            current_monthly_social_security_nominal = current_social_security_nominal / 12.0

        net_withdrawal_nominal = current_monthly_spending_nominal
        if include_social_security and (age_in_months / 12.0) >= social_security_start_age:
            net_withdrawal_nominal = max(0.0, net_withdrawal_nominal -
                                        current_monthly_social_security_nominal)

        portfolio -= net_withdrawal_nominal
        if portfolio <= 0:
            return False, 0.0, 0.0, 0.0, []

        # Get market return from bootstrap or parametric model
        if use_bootstrap and bootstrap_returns is not None and bootstrap_month_idx < len(bootstrap_returns):
            market_return = bootstrap_returns[bootstrap_month_idx]
            bootstrap_month_idx += 1
        else:
            # Fall back to parametric model if bootstrap fails or runs out
            market_return, current_variance = simulate_monthly_return_svj(
                rng_local, params_annual, current_variance)
        
        portfolio *= (1.0 + market_return)
        portfolio_history.append(portfolio)

        age_in_months += 1

    growth_rates = [portfolio_history[i] / portfolio_history[i-1] - 1.0
                   for i in range(1, len(portfolio_history))]
    mean_growth = np.mean(growth_rates) if growth_rates else 0.0
    std_dev = np.std(growth_rates) if growth_rates else 0.0
    max_drawdown = calculate_max_drawdown(np.array(portfolio_history))

    return True, mean_growth, std_dev, max_drawdown, portfolio_history


def check_success_rate_worker(principal, retirement_age, num_sims, seed_offset,
                              config, params, bootstrap_data=None):
    if config.seed is None:
        nested_rng = np.random.default_rng()
    else:
        nested_rng = np.random.default_rng(seed=(config.seed + seed_offset + 1))

    # Create bootstrap sampler for this worker if block bootstrap is enabled
    bootstrap_sampler = None
    if config.use_block_bootstrap:
        try:
            monthly_returns = None
            monthly_inflation = None
            if bootstrap_data is not None:
                monthly_returns, monthly_inflation = bootstrap_data
            bootstrap_sampler = create_block_bootstrap_sampler(
                config, nested_rng, monthly_returns, monthly_inflation)
        except Exception as e:
            logger.error(f"Worker {seed_offset} failed to create bootstrap sampler: {e}")
            # Fall back to parametric model
            bootstrap_sampler = None

    successes = 0
    metrics = {'mean_growth': [], 'std_dev': [], 'max_drawdown': []}

    for _ in range(num_sims):
        is_success, mg, sd, mdd, _ = simulate_withdrawals(
            principal, retirement_age, nested_rng, params, config.spending_real,
            config.include_social_security, config.social_security_real,
            config.social_security_start_age, config, bootstrap_sampler
        )
        if is_success:
            successes += 1
            metrics['mean_growth'].append(mg)
            metrics['std_dev'].append(sd)
            metrics['max_drawdown'].append(mdd)

    return {'successes': successes, 'metrics': metrics}


def check_success_rate(principal, retirement_age, num_nested_sims, config, params, bootstrap_data=None):
    # CPU-based implementation
    if config.num_workers <= 1 or num_nested_sims < 100:
        res = check_success_rate_worker(principal, retirement_age, num_nested_sims,
                                       0, config, params, bootstrap_data)
        success_rate = res['successes'] / max(1, num_nested_sims)
        combined_metrics = {k: np.array(v) for k, v in res['metrics'].items()}
        return success_rate, combined_metrics

    sims_per_worker = num_nested_sims // config.num_workers
    remaining = num_nested_sims % config.num_workers
    futures = []

    # Note: For multiprocessing, bootstrap_data can't be passed directly (pickling issues)
    # Each worker process will load from cache (loaded once per process)
    with ProcessPoolExecutor(max_workers=config.num_workers) as executor:
        for i in range(config.num_workers):
            sims_this_worker = sims_per_worker + (1 if i < remaining else 0)
            if sims_this_worker > 0:
                futures.append(executor.submit(check_success_rate_worker,
                                             principal, retirement_age,
                                             sims_this_worker, i, config, params, None))

        results = [f.result() for f in futures]

    total_successes = sum(r['successes'] for r in results)
    all_mg = [np.array(r['metrics']['mean_growth']) for r in results
              if r['metrics']['mean_growth']]
    all_sd = [np.array(r['metrics']['std_dev']) for r in results
              if r['metrics']['std_dev']]
    all_mdd = [np.array(r['metrics']['max_drawdown']) for r in results
               if r['metrics']['max_drawdown']]

    combined_metrics = {
        'mean_growth': np.concatenate(all_mg) if all_mg else np.array([]),
        'std_dev': np.concatenate(all_sd) if all_sd else np.array([]),
        'max_drawdown': np.concatenate(all_mdd) if all_mdd else np.array([]),
    }

    success_rate = total_successes / max(1, num_nested_sims)
    return success_rate, combined_metrics


def find_required_principal(target_age, success_target, num_nested_sims, config, params,
                            warm_start_principal=None, bootstrap_data=None):
    """Find required principal with optional warm start from previous age.
    
    Args:
        warm_start_principal: Previous age's principal to use as starting point for faster convergence
    """
    # Use warm start to narrow the search range significantly
    if warm_start_principal is not None and warm_start_principal > 0:
        # Start search range around warm start (wider range for safety)
        # Use ±50% range around warm start, with minimum bounds
        search_range_factor = 0.5
        low_principal = max(10_000.0, warm_start_principal * (1 - search_range_factor))
        high_principal = min(20_000_000.0, warm_start_principal * (1 + search_range_factor))
        
        # Ensure range is wide enough for binary search
        min_range = warm_start_principal * 0.2  # At least 20% range
        current_range = high_principal - low_principal
        if current_range < min_range:
            center = (low_principal + high_principal) / 2.0
            low_principal = max(10_000.0, center - min_range / 2.0)
            high_principal = min(20_000_000.0, center + min_range / 2.0)
    else:
        # Default wide search range
        low_principal = 10_000.0
        high_principal = 20_000_000.0
    
    tolerance = 1000.0
    principal_cache = {}
    max_iterations = 30  # Safety limit
    
    iteration = 0
    while high_principal - low_principal > tolerance and iteration < max_iterations:
        mid_principal = (low_principal + high_principal) / 2.0
        cache_key = round(mid_principal, 2)

        if cache_key in principal_cache:
            success_rate = principal_cache[cache_key]
        else:
            success_rate, _ = check_success_rate(mid_principal, target_age,
                                                 num_nested_sims, config, params, bootstrap_data)
            principal_cache[cache_key] = success_rate

        if success_rate >= success_target:
            high_principal = mid_principal
        else:
            low_principal = mid_principal
        
        iteration += 1

    return high_principal


# ============================================================================
# ACCUMULATION SIMULATION
# ============================================================================

def create_simulation_record(sim, age_in_months, is_retired, portfolio,
                            current_variance, principal_lookup, current_age_years,
                            current_monthly_spending_nominal, current_annual_ss_nominal,
                            annual_inflation_draw, cumulative_inflation_since_start,
                            savings_rate_for_month, current_monthly_income_real,
                            is_unemployed, dollars_saved_nominal,
                            monthly_salary_growth_rate, market_return,
                            portfolio_growth_factor, config):
    return {
        'SIM_ID': sim,
        'AGE': age_in_months / 12.0,
        'RETIRED?': is_retired,
        'PORTFOLIO_VALUE': portfolio,
        'VOLATILITY': current_variance,
        'REQUIRED_REAL_PRINCIPAL': principal_lookup.get(
            current_age_years, {}).get('principal_real', np.nan),
        'WITHDRAWAL_RATE': principal_lookup.get(
            current_age_years, {}).get('swr', np.nan),
        'REQUIRED_NOMINAL_PRINCIPAL': principal_lookup.get(
            current_age_years, {}).get('principal_nominal', np.nan),
        'NOMINAL_DESIRED_CONSUMPTION': current_monthly_spending_nominal * 12.0,
        'REAL_DESIRED_CONSUMPTION': config.spending_real,
        'ANNUAL_INFLATION': (annual_inflation_draw
                            if (age_in_months % 12) == 0 else np.nan),
        'CUMULATIVE_INFLATION': cumulative_inflation_since_start,
        'REAL_SOCIAL_SECURITY_BENEFIT': (
            config.social_security_real
            if is_retired and current_age_years >= config.social_security_start_age
            else np.nan),
        'NOMINAL_SOCIAL_SECURITY_BENEFIT': (
            current_annual_ss_nominal
            if is_retired and current_age_years >= config.social_security_start_age
            else np.nan),
        'SAVINGS_RATE': savings_rate_for_month * 12.0 if not is_retired else np.nan,
        'SALARY_REAL': (current_monthly_income_real * 12.0
                       if not is_retired else np.nan),
        'SALARY_NOMINAL': (
            (current_monthly_income_real * cumulative_inflation_since_start) * 12.0
            if not is_retired else np.nan),
        'EMPLOYED?': (not is_unemployed) if not is_retired else False,
        'DOLLARS_SAVED': dollars_saved_nominal * 12.0 if not is_retired else np.nan,
        'REAL_SALARY_GROWTH_RATE': (
            (monthly_salary_growth_rate * 12.0) if not is_retired else np.nan),
        'NOMINAL_SALARY_GROWTH_RATE': (
            ((1 + monthly_salary_growth_rate) * (1 + annual_inflation_draw) - 1.0) * 12.0
            if not is_retired else np.nan),
        'MONTHLY_PORTFOLIO_RETURN': market_return,
        'CUMULATIVE_PORTFOLIO_RETURN': portfolio_growth_factor - 1.0,
    }


def run_single_accumulation_simulation(sim, config, params, principal_lookup, rng,
                                       unemp_config, monthly_income_real,
                                       monthly_salary_growth_rate,
                                       monthly_salary_growth_rate_std_dev,
                                       bootstrap_data=None):
    portfolio = float(config.initial_portfolio)
    age_in_months = int(config.initial_age * 12)
    retirement_age = np.nan
    is_retired = False
    portfolio_growth_factor = 1.0
    current_sim_record = []
    # Initialize variance for parametric model (always initialize, even if using bootstrap as fallback)
    current_variance = params["v0"]
    cumulative_inflation_since_start = 1.0
    annual_inflation_draw = 0.0
    current_monthly_income_real = monthly_income_real

    unemployment = Unemployment(config=unemp_config, rng_local=rng)

    current_annual_spending_nominal = config.spending_real
    current_monthly_spending_nominal = current_annual_spending_nominal / 12.0
    current_annual_ss_nominal = config.social_security_real
    current_monthly_ss_nominal = current_annual_ss_nominal / 12.0

    # Pre-sample all returns and inflation if using block bootstrap
    use_bootstrap = False
    if config.use_block_bootstrap:
        try:
            monthly_returns = None
            monthly_inflation = None
            if bootstrap_data is not None:
                monthly_returns, monthly_inflation = bootstrap_data
            bootstrap_sampler = create_block_bootstrap_sampler(
                config, rng, monthly_returns, monthly_inflation)
            total_months = int((config.death_age - config.initial_age) * 12)
            bootstrap_returns, bootstrap_inflation = bootstrap_sampler.sample_sequence(total_months)
            bootstrap_month_idx = 0
            use_bootstrap = True
        except Exception as e:
            logger.warning(f"Simulation {sim} failed to create bootstrap sampler, falling back to parametric model: {e}")
            bootstrap_returns = None
            bootstrap_inflation = None
            bootstrap_month_idx = None
            use_bootstrap = False
    else:
        bootstrap_returns = None
        bootstrap_inflation = None
        bootstrap_month_idx = None
        use_bootstrap = False

    while (age_in_months / 12.0) <= config.death_age:
        current_age_years = int(age_in_months // 12)

        if (age_in_months % 12) == 0:
            if (not is_retired) and (current_age_years in principal_lookup):
                req = principal_lookup[current_age_years]
                required_principal_real = req.get('principal_real', np.nan)
                if not np.isnan(required_principal_real):
                    required_principal_nominal = (required_principal_real *
                                                 cumulative_inflation_since_start)
                    if portfolio >= required_principal_nominal:
                        retirement_age = current_age_years
                        is_retired = True

        if (age_in_months > int(config.initial_age * 12)) and ((age_in_months % 12) == 0):
            if use_bootstrap and bootstrap_inflation is not None:
                # Calculate annual inflation from the previous 12 months of bootstrap inflation
                # At the start of a new year, bootstrap_month_idx represents months already processed
                # So we calculate inflation from the previous 12 months (indices bootstrap_month_idx-12 to bootstrap_month_idx-1)
                # which corresponds to bootstrap_inflation[bootstrap_month_idx-12:bootstrap_month_idx]
                year_start_idx = max(0, bootstrap_month_idx - 12)
                year_end_idx = bootstrap_month_idx
                if year_end_idx > year_start_idx and year_end_idx - year_start_idx == 12:
                    monthly_inflations = bootstrap_inflation[year_start_idx:year_end_idx]
                    annual_inflation_draw = np.prod(1.0 + monthly_inflations) - 1.0
                    # Inflation from bootstrap data is being used (synchronized with returns from same blocks)
                else:
                    # Fallback if we don't have enough data yet
                    annual_inflation_draw = rng.normal(config.mean_inflation_geometric,
                                                       config.std_inflation)
            else:
                annual_inflation_draw = rng.normal(config.mean_inflation_geometric,
                                                  config.std_inflation)
            annual_inflation_draw = max(annual_inflation_draw, -0.99)
            cumulative_inflation_since_start *= (1.0 + annual_inflation_draw)

            current_annual_spending_nominal *= (1.0 + annual_inflation_draw)
            current_monthly_spending_nominal = current_annual_spending_nominal / 12.0
            current_annual_ss_nominal *= (1.0 + annual_inflation_draw)
            current_monthly_ss_nominal = current_annual_ss_nominal / 12.0

        # Get market return from bootstrap or parametric model
        if use_bootstrap and bootstrap_returns is not None and bootstrap_month_idx < len(bootstrap_returns):
            market_return = bootstrap_returns[bootstrap_month_idx]
            bootstrap_month_idx += 1
        else:
            # Fall back to parametric model if bootstrap fails or runs out
            market_return, current_variance = simulate_monthly_return_svj(
                rng, params, current_variance)
        portfolio_growth_factor *= (1.0 + market_return)

        dollars_saved_nominal = 0.0
        savings_rate_for_month = 0.0
        is_unemployed = False

        if not is_retired:
            is_unemployed = unemployment.update(monthly_return=market_return)

            savings_rate_for_month = (config.savings_rate if not is_unemployed
                                     else config.unemployment_savings_rate)
            savings_rate_for_month = max(0.0, savings_rate_for_month)

            income_for_month_real = current_monthly_income_real * (
                config.unemployment_income_multiplier if is_unemployed else 1.0)
            income_for_month_nominal = income_for_month_real * cumulative_inflation_since_start

            dollars_saved_nominal = income_for_month_nominal * savings_rate_for_month
            portfolio += dollars_saved_nominal

            salary_growth_rate_this_month_real = rng.normal(
                monthly_salary_growth_rate, monthly_salary_growth_rate_std_dev)
            current_monthly_income_real = max(0.0, current_monthly_income_real *
                                             (1.0 + salary_growth_rate_this_month_real))
        else:
            net_withdrawal = current_monthly_spending_nominal
            if (config.include_social_security and
                current_age_years >= config.social_security_start_age):
                net_withdrawal = max(0.0, net_withdrawal - current_monthly_ss_nominal)
            portfolio -= net_withdrawal

        portfolio *= (1.0 + market_return)

        if is_retired and portfolio <= 0.0:
            portfolio = 0.0

        if portfolio <= 0.0:
            break

        if sim < config.num_sims_to_export:
            record_dict = create_simulation_record(
                sim, age_in_months, is_retired, portfolio, current_variance,
                principal_lookup, current_age_years, current_monthly_spending_nominal,
                current_annual_ss_nominal, annual_inflation_draw,
                cumulative_inflation_since_start, savings_rate_for_month,
                current_monthly_income_real, is_unemployed, dollars_saved_nominal,
                monthly_salary_growth_rate, market_return, portfolio_growth_factor,
                config
            )
            current_sim_record.append(record_dict)

        age_in_months += 1

    final_bequest_nominal = portfolio
    final_bequest_real = final_bequest_nominal / cumulative_inflation_since_start

    return {
        'retirement_age': retirement_age,
        'final_bequest_nominal': final_bequest_nominal,
        'final_bequest_real': final_bequest_real,
        'simulation_record': current_sim_record
    }


def run_accumulation_simulations(config, params, principal_lookup, rng, bootstrap_data=None):
    retirement_ages = np.full(config.num_outer, np.nan)
    ever_retired = np.zeros(config.num_outer, dtype=bool)
    detailed_simulations_to_export = []
    all_final_bequest_nominal = []
    all_final_bequest_real = []

    unemp_config = UnemploymentConfig(
        baseline_exit=0.25,
        entry_prob_annual=config.unemployment_prob,
        lag_months=12,
        rho=-0.25
    )

    monthly_income_real = config.annual_income_real / 12.0
    monthly_salary_growth_rate = (1 + config.annual_salary_growth_rate)**(1/12.0) - 1.0
    monthly_salary_growth_rate_std_dev = config.salary_growth_rate_std_dev / np.sqrt(12.0)

    for sim in tqdm(range(config.num_outer), desc="Running Simulations"):
        result = run_single_accumulation_simulation(
            sim, config, params, principal_lookup, rng, unemp_config,
            monthly_income_real, monthly_salary_growth_rate,
            monthly_salary_growth_rate_std_dev, bootstrap_data
        )

        retirement_ages[sim] = result['retirement_age']
        if not np.isnan(result['retirement_age']):
            ever_retired[sim] = True

        all_final_bequest_nominal.append(result['final_bequest_nominal'])
        all_final_bequest_real.append(result['final_bequest_real'])

        if sim < config.num_sims_to_export:
            detailed_simulations_to_export.append(result['simulation_record'])

    return (retirement_ages, ever_retired, detailed_simulations_to_export,
            all_final_bequest_nominal, all_final_bequest_real)


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_required_principal_nominal(ages, principals_nominal, swr, config, median_age):
    if not HAS_MATPLOTLIB:
        return

    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax1.set_xlabel('Retirement Age', color='white', fontsize=12)
    ax1.set_ylabel('Required Principal ($)', color='white', fontsize=12)
    ax1.plot(ages, principals_nominal, color='cyan', marker='o',
            label='Required Principal (Nominal $)', linewidth=2)
    ax1.tick_params(axis='y', labelcolor='white')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.get_yaxis().set_major_formatter(
        plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    ax2 = ax1.twinx()
    ax2.set_ylabel('Withdrawal Rate (%)', color='white', fontsize=12)
    ax2.plot(ages, np.array(swr), color='magenta', marker='x',
            linestyle='--', label='Withdrawal Rate', linewidth=2)
    ax2.tick_params(axis='y', labelcolor='white')
    ax2.get_yaxis().set_major_formatter(
        plt.FuncFormatter(lambda x, p: f'{x:.2f}%'))

    if config.include_social_security:
        ax1.axvline(x=config.social_security_start_age, color='lime',
                   linestyle=':',
                   label=f'Social Security (Age {config.social_security_start_age})',
                   linewidth=2)

    if not np.isnan(median_age):
        ax1.axvline(x=median_age, color='yellow', linestyle='--',
                   label=f'Median Retirement Age ({median_age:.1f})', linewidth=2)

    success_pct = int(config.success_target * 100)
    fig.suptitle(f"Required Principal & Withdrawal Rate for {success_pct}% Success (Nominal)",
                fontsize=14, color='white')
    fig.tight_layout()
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    import os
    os.makedirs(config.output_directory, exist_ok=True)
    filepath = os.path.join(config.output_directory, 'required_principal_and_swr_nominal.png')
    plt.savefig(filepath, bbox_inches='tight', format='png', dpi=150)
    plt.show()
    plt.close(fig)


def plot_required_principal_real(ages, principals_real, swr, config, median_age):
    if not HAS_MATPLOTLIB:
        return

    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax1.set_xlabel('Retirement Age', color='white', fontsize=12)
    ax1.set_ylabel('Required Principal ($)', color='white', fontsize=12)
    ax1.plot(ages, principals_real, color='orange', marker='o',
            label='Required Principal (Real $)', linewidth=2)
    ax1.tick_params(axis='y', labelcolor='white')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.get_yaxis().set_major_formatter(
        plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    ax2 = ax1.twinx()
    ax2.set_ylabel('Withdrawal Rate (%)', color='white', fontsize=12)
    ax2.plot(ages, np.array(swr), color='lime', marker='x',
            linestyle='--', label='Withdrawal Rate', linewidth=2)
    ax2.tick_params(axis='y', labelcolor='white')
    ax2.get_yaxis().set_major_formatter(
        plt.FuncFormatter(lambda x, p: f'{x:.2f}%'))

    if config.include_social_security:
        ax1.axvline(x=config.social_security_start_age, color='cyan',
                   linestyle=':',
                   label=f'Social Security (Age {config.social_security_start_age})',
                   linewidth=2)

    if not np.isnan(median_age):
        ax1.axvline(x=median_age, color='yellow', linestyle='--',
                   label=f'Median Retirement Age ({median_age:.1f})', linewidth=2)

    success_pct = int(config.success_target * 100)
    fig.suptitle(f"Required Principal & Withdrawal Rate for {success_pct}% Success (Real)",
                fontsize=14, color='white')
    fig.tight_layout()
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    import os
    os.makedirs(config.output_directory, exist_ok=True)
    filepath = os.path.join(config.output_directory, 'required_principal_and_swr_real.png')
    plt.savefig(filepath, bbox_inches='tight', format='png', dpi=150)
    plt.show()
    plt.close(fig)


def plot_retirement_age_distribution(valid_ages, median_age, output_dir='output'):
    if not HAS_MATPLOTLIB or valid_ages.size == 0:
        return

    plt.figure(figsize=(12, 7))
    plt.hist(valid_ages, bins=20, color='cyan', edgecolor='black', alpha=0.7)
    plt.axvline(median_age, color='red', linestyle='--', linewidth=2,
               label=f'Median: {median_age:.1f}')
    plt.title('Distribution of Retirement Ages', fontsize=14, color='white')
    plt.xlabel('Retirement Age', color='white', fontsize=12)
    plt.ylabel('Frequency', color='white', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    import os
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'retirement_age_distribution.png')
    plt.savefig(filepath, bbox_inches='tight', format='png', dpi=150)
    plt.show()
    plt.close()


def plot_cumulative_retirement_probability(retirement_ages, config, median_age, num_outer):
    if not HAS_MATPLOTLIB:
        return

    valid_retirement_ages = retirement_ages[~np.isnan(retirement_ages)]
    if valid_retirement_ages.size == 0:
        return

    sorted_ages = np.sort(valid_retirement_ages)
    cumulative_prob = np.arange(1, len(sorted_ages) + 1) / num_outer * 100

    plt.figure(figsize=(12, 7))
    plt.plot(sorted_ages, cumulative_prob, color='lime', marker='o',
            linestyle='-', markersize=4, alpha=0.8, linewidth=2)

    if config.include_social_security:
        plt.axvline(x=config.social_security_start_age, color='white',
                   linestyle=':',
                   label=f'Social Security (Age {config.social_security_start_age})',
                   linewidth=2)

    if not np.isnan(median_age):
        plt.axvline(x=median_age, color='white', linestyle='--',
                   label=f'Median Retirement Age ({median_age:.1f})', linewidth=2)

    plt.title("Cumulative Probability of Retiring by Age", fontsize=14, color='white')
    plt.xlabel("Age", color='white', fontsize=12)
    plt.ylabel("Cumulative Probability (%)", color='white', fontsize=12)
    plt.ylim(0, 100)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    plt.legend(facecolor='black', edgecolor='white', framealpha=0.6)
    plt.tight_layout()
    import os
    os.makedirs(config.output_directory, exist_ok=True)
    filepath = os.path.join(config.output_directory, 'cumulative_prob_retiring_by_age.png')
    plt.savefig(filepath, bbox_inches='tight', format='png', dpi=150)
    plt.show()
    plt.close()


def create_all_plots(required_principal_data, retirement_ages,
                     detailed_simulations, config, median_age, num_outer):
    ages = [row['age'] for row in required_principal_data]
    principals_nominal = [row['principal_nominal'] for row in required_principal_data]
    principals_real = [row['principal_real'] for row in required_principal_data]
    swr = [row['swr'] for row in required_principal_data]

    plot_required_principal_nominal(ages, principals_nominal, swr, config, median_age)
    plot_required_principal_real(ages, principals_real, swr, config, median_age)

    valid_ages = retirement_ages[~np.isnan(retirement_ages)]
    if valid_ages.size > 0:
        plot_retirement_age_distribution(valid_ages, median_age, config.output_directory)

    plot_cumulative_retirement_probability(retirement_ages, config, median_age,
                                          num_outer)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def build_required_principal_table(config, params, bootstrap_data=None):
    success_pct_label = f"{int(config.success_target * 100)}%"
    print(f"\n--- Stage 1: Building Required Principal Lookup Table "
          f"({success_pct_label} success) ---")

    target_ages = np.arange(30, 71)
    required_principal_table = {}
    mean_inflation_arithmetic = convert_geometric_to_arithmetic(
        config.mean_inflation_geometric, config.std_inflation)

    # Use warm start from previous age to significantly speed up binary search
    # This reduces binary search iterations from ~15-20 to ~5-8 iterations per age
    previous_principal = None
    for age in tqdm(target_ages, desc="Calculating required principal per age"):
        principal = find_required_principal(age, config.success_target,
                                           config.num_nested, config, params,
                                           warm_start_principal=previous_principal,
                                           bootstrap_data=bootstrap_data)
        
        # Enforce deviation threshold if enabled
        if config.use_principal_deviation_threshold and previous_principal is not None:
            # Calculate the maximum allowed change
            max_change = previous_principal * config.principal_deviation_threshold
            min_allowed = previous_principal - max_change
            max_allowed = previous_principal + max_change
            
            # Constrain the principal to within the threshold
            if principal < min_allowed:
                principal = min_allowed
                logger.info(
                    f"Age {age}: Principal constrained to minimum allowed "
                    f"(${principal:,.2f} due to {config.principal_deviation_threshold*100:.2f}% threshold)"
                )
            elif principal > max_allowed:
                principal = max_allowed
                logger.info(
                    f"Age {age}: Principal constrained to maximum allowed "
                    f"(${principal:,.2f} due to {config.principal_deviation_threshold*100:.2f}% threshold)"
                )
        
        required_principal_table[int(age)] = principal
        previous_principal = principal

    required_principal_data = []
    for age, principal_real in required_principal_table.items():
        principal_nominal = calculate_nominal_value(
            principal_real, config.initial_age, age, mean_inflation_arithmetic)

        net_withdrawal_real = config.spending_real
        if config.include_social_security and age >= config.social_security_start_age:
            net_withdrawal_real = max(0.0, config.spending_real -
                                     config.social_security_real)

        swr_val = ((net_withdrawal_real / principal_real) * 100.0
                  if principal_real > 0 else np.nan)

        nominal_spending = calculate_nominal_value(
            config.spending_real, config.initial_age, age, mean_inflation_arithmetic)
        nominal_ss = 0.0
        if config.include_social_security:
            nominal_ss = calculate_nominal_value(
                config.social_security_real, config.initial_age, age,
                mean_inflation_arithmetic)

        net_withdrawal_nominal = nominal_spending
        if config.include_social_security and age >= config.social_security_start_age:
            net_withdrawal_nominal = max(0.0, nominal_spending - nominal_ss)

        required_principal_data.append({
            'age': age,
            'principal_real': principal_real,
            'principal_nominal': principal_nominal,
            'spending_real': config.spending_real,
            'spending_nominal': nominal_spending,
            'net_withdrawal_real': net_withdrawal_real,
            'net_withdrawal_nominal': net_withdrawal_nominal,
            'swr': swr_val
        })

    if config.generate_csv_summary:
        export_to_csv(required_principal_data, 'required_principal_table.csv', config.output_directory)

    display_principal_table(required_principal_data, success_pct_label)

    principal_lookup = {
        int(row['age']): {
            'principal_real': row['principal_real'],
            'principal_nominal': row['principal_nominal'],
            'swr': row['swr']
        } for row in required_principal_data
    }

    return required_principal_data, principal_lookup


def display_principal_table(required_principal_data, success_pct_label):
    df_table = pd.DataFrame(required_principal_data)
    df_table_display = df_table.copy()

    for col in ['principal_real', 'principal_nominal', 'spending_real',
                'spending_nominal', 'net_withdrawal_real', 'net_withdrawal_nominal']:
        df_table_display[col] = df_table_display[col].apply(
            lambda x: f"${x:,.2f}")

    df_table_display['swr'] = df_table_display['swr'].apply(
        lambda x: f"{x:.2f}%" if not np.isnan(x) else "NaN")

    df_table_display.rename(columns={
        'age': 'Retirement Age',
        'principal_real': 'Principal (Real $)',
        'principal_nominal': 'Principal (Nominal $)',
        'spending_real': 'Spending (Real $)',
        'spending_nominal': 'Spending (Nominal $)',
        'net_withdrawal_real': 'Net Withdrawal (Real $)',
        'net_withdrawal_nominal': 'Net Withdrawal (Nominal $)',
        'swr': 'Withdrawal Rate'
    }, inplace=True)

    print_rich_table(df_table_display,
                    f"Required Principal & Withdrawal Rate for {success_pct_label} "
                    f"Success (Real principals)")


def display_final_results(retirement_ages, config):
    print("\n--- Final Results ---")

    valid_ages = retirement_ages[~np.isnan(retirement_ages)]
    num_retired = len(valid_ages)
    pct_ever_retired = 100.0 * num_retired / max(1, config.num_outer)

    median_age = np.nan if valid_ages.size == 0 else np.median(valid_ages)
    p10 = np.nan if valid_ages.size == 0 else np.percentile(valid_ages, 10)
    p25 = np.nan if valid_ages.size == 0 else np.percentile(valid_ages, 25)
    p75 = np.nan if valid_ages.size == 0 else np.percentile(valid_ages, 75)
    p90 = np.nan if valid_ages.size == 0 else np.percentile(valid_ages, 90)

    def prob_retire_before(age_limit):
        if valid_ages.size == 0:
            return 0.0
        return 100.0 * np.sum(valid_ages <= age_limit) / config.num_outer

    prob_before_50 = prob_retire_before(50)
    prob_before_55 = prob_retire_before(55)
    prob_before_60 = prob_retire_before(60)

    print(f"\nSimulations run: {config.num_outer}")
    print(f"Ever retire with >= {config.success_target*100:.0f}% success: "
          f"{pct_ever_retired:.2f}%")
    print(f"Median retirement age: {median_age:.1f}")
    print(f"10th percentile retirement age: {p10:.1f}")
    print(f"25th percentile retirement age: {p25:.1f}")
    print(f"75th percentile retirement age: {p75:.1f}")
    print(f"90th percentile retirement age: {p90:.1f}")
    print(f"Probability retire before age 50: {prob_before_50:.2f}%")
    print(f"Probability retire before age 55: {prob_before_55:.2f}%")
    print(f"Probability retire before age 60: {prob_before_60:.2f}%")

    return median_age


def main():
    """Main execution function - run this to start the simulation"""
    print("\n" + "="*70)
    print("CYTHONIZED RETIREMENT SIMULATION v6.3")
    print("="*70)

    # Print Cython status (only once, in main)
    if CYTHON_AVAILABLE:
        print("[OK] Cython module imported successfully (compiled extension)")
        print("[OK] Running with Cython acceleration (10-50x faster!)")
    else:
        print("[WARNING] Running in pure Python mode (no Cython acceleration)")
        print("  To enable Cython acceleration, compile the module:")
        print("  Run: cython/build_cython_fixed.bat (or python setup.py build_ext --inplace)")

    print("="*70 + "\n")

    try:
        config = SimulationConfig()
        config.validate()
        params = config.params

        # Log block bootstrap configuration
        if config.use_block_bootstrap:
            print(f"[INFO] Block Bootstrap ENABLED")
            print(f"  CSV Path: {config.bootstrap_csv_path}")
            print(f"  Portfolio Column: {config.portfolio_column_name}")
            print(f"  Block Length: {config.block_length_years} years ({config.block_length_years * 12} months)")
            print(f"  Block Type: {'Overlapping' if config.block_overlapping else 'Non-overlapping'}")
            print(f"  Note: A new random {config.block_length_years}-year block is sampled every {config.block_length_years} years")
            print(f"       (e.g., an 80-year simulation uses {int(80/config.block_length_years)} different random blocks)")
        else:
            print(f"[INFO] Using Parametric Model (Bates/Heston)")
            print(f"  Model Parameters: mu={params['mu']:.4f}, kappa={params['kappa']:.4f}, "
                  f"theta={params['theta']:.4f}, nu={params['nu']:.4f}")
        print()

        # Load bootstrap data once if block bootstrap is enabled
        bootstrap_data = None
        if config.use_block_bootstrap:
            print("[INFO] Loading bootstrap data from CSV (one-time operation)...")
            bootstrap_data = load_bootstrap_data(config)
            if bootstrap_data is None:
                logger.warning("Failed to load bootstrap data, falling back to parametric model")
                config.use_block_bootstrap = False
            else:
                print(f"[OK] Bootstrap data loaded: {len(bootstrap_data[0])} monthly returns")
        print()

        if config.seed is not None:
            rng = np.random.default_rng(seed=config.seed)
        else:
            rng = np.random.default_rng()

        required_principal_data, principal_lookup = build_required_principal_table(
            config, params, bootstrap_data)

        print("\n--- Stage 2: Running Accumulation Simulations "
              "(monthly returns, annual inflation adjustments) ---")

        (retirement_ages, ever_retired, detailed_simulations,
         final_bequest_nominal, final_bequest_real) = run_accumulation_simulations(
            config, params, principal_lookup, rng, bootstrap_data)

        median_age = display_final_results(retirement_ages, config)

        if config.generate_csv_summary:
            export_detailed_simulations_to_csv(
                detailed_simulations, 'detailed_lifecycle_paths.csv', config.output_directory)

        create_all_plots(required_principal_data, retirement_ages,
                        detailed_simulations, config, median_age, config.num_outer)

        logger.info("[OK] Simulation completed successfully")

        if CYTHON_AVAILABLE:
            print("\n[SUCCESS] Cython acceleration was used - simulation ran much faster!")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# RUN THE SIMULATION
# ============================================================================

if __name__ == "__main__":
    main()

# For Colab: Uncomment the line below to run automatically
# main()