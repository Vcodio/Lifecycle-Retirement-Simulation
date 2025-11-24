# ==============================================================
# VERSION 2.0 — Hidden Markov Model Regime Detection (Volatility & Skewness)
# ==============================================================
# This script infers market regimes using both volatility and skewness
# from historical asset prices using a Hidden Markov Model (HMM).
# It handles data preprocessing, feature engineering, and visualization.
#
# Key Features:
# - Stage 1: Cleans and prepares historical data, handling malformed
#            files and missing values robustly.
# - Stage 2: Generates volatility and skewness features for HMM input (2D feature space).
# - Stage 3: Fits a Gaussian HMM to monthly volatility and 12-month skewness
#            and infers latent market regimes.
# - Stage 4: Dynamically maps HMM states based on volatility and skewness characteristics.
# - Stage 5: Calculates regime metrics including transition matrices,
#            mean durations, stationary probabilities, and risk statistics.
# - Stage 6: Visualizes regimes and overlays daily real returns.
# - Stage 7: Exports monthly regime classification with nominal returns
#            for further analysis.
#
# NEW FEATURES IN VERSION 2.0:
# - Multi-regime HMM with 2D feature space (volatility + skewness).
# - Forward-fill mapping of monthly regimes to daily data for risk analysis.
# - Robust file parsing for CSV or fixed-width data.
# - Optional start-date filtering for analysis subset.
# - Debug mode with enhanced console output using Rich.
# - Derived metrics: annual real/nominal returns, max drawdown.
# - Comprehensive visualization with regime shading.
#
# Dependencies:
#   pip install numpy pandas hmmlearn matplotlib rich scipy
#
# Configuration Options:
# - NUM_REGIMES: Number of HMM latent states (default: 3)
# - LOOKBACK_PERIOD: Non-overlapping monthly step (default: 21 trading days)
# - VOLATILITY_WINDOW: Rolling window for volatility calculation (default: 21 trading days)
# - SKEWNESS_WINDOW: Rolling window for skewness calculation (default: 252 trading days = 12 months)
# - RANDOM_SEED: Seed for reproducibility (default: 42)
# - DEBUG_MODE: Enables detailed intermediate output
# - START_DATE_ANALYSIS: Optional start date for filtering historical data
# - ASSET_PRICE_COLUMN_NAME: Column name of asset prices in CSV
# --------------------------------------------------------------------------
# Author: VCODIO
# --------------------------------------------------------------------------

import numpy as np
import pandas as pd
from hmmlearn import hmm
import matplotlib.pyplot as plt
import math
from scipy.stats import norm
# NOTE: Requires 'pip install rich' for enhanced console output
from rich.console import Console
from rich.table import Table

# --- 1. CONFIGURATION ---
NUM_REGIMES = 6  # Number of regimes (will be characterized by vol and skew combinations)
# The lookback period for non-overlapping steps (21 days is approx 1 trading month)
LOOKBACK_PERIOD = 21
# Rolling window for volatility calculation (21 trading days = 1 month)
VOLATILITY_WINDOW = 21
# Rolling window for skewness calculation (252 trading days = 12 months)
SKEWNESS_WINDOW = 252
RANDOM_SEED = 1
# NEW: Flag to enable verbose debugging output
DEBUG_MODE = True
# NEW: Flag to enable/disable final plot generation
ENABLE_PLOT = True

# NEW FEATURE: Select the start date for the analysis (inclusive).
# Set to None to use all available data after lookback/cleaning. Format: 'YYYY-MM-DD'
START_DATE_ANALYSIS = '1800-01-01'

# NEW: Define the name of the column containing the price data for the asset/portfolio.
# If you change your CSV column name, update this variable.
ASSET_PRICE_COLUMN_NAME = 'S&P 500'

# We set the random_state on the HMM model explicitly for stability.
np.random.seed(RANDOM_SEED)
TRADING_DAYS_YEAR = 252 # Used for annualizing volatility and returns
ANNUALIZATION_FACTOR = 12 # 12 months per year

# Regime labels will be assigned based on both volatility and skewness characteristics
REGIME_LABELS = {} # This will store the final HMM Index (e.g., R1) -> Canonical Label mapping

# Initialize Rich Console for all output
console = Console()

# Helper function for Max Drawdown calculation
def calculate_max_drawdown(daily_log_returns_series):
    """Calculates Max Drawdown from a series of daily log returns (in percent)."""
    if daily_log_returns_series.empty:
        return 0.0
    # Convert log returns (%) to simple returns (ratio)
    # The log return is in percent, so divide by 100 before taking exp/subtracting 1
    simple_returns = np.exp(daily_log_returns_series / 100) - 1
    # Calculate cumulative simple returns
    cumulative_returns = (1 + simple_returns).cumprod()
    # Calculate running maximum (high water mark)
    running_max = cumulative_returns.cummax()
    # Calculate drawdown
    drawdown = (cumulative_returns / running_max) - 1
    # Max drawdown is the minimum value (most negative)
    return drawdown.min() * 100

# --- 2. DATA LOADING & PREPROCESSING ---
console.print("\n--- 2. Loading and Preprocessing Historical Data ---", style="bold")

# --- ACTUAL DATA LOADING (Highly Robustified for Delimited and FWF) ---
file_path = "data/Market and Inflation - Regime Testing.csv"
df = None
loaded = False

# Attempt 1: Delimited (Comma, Tab, Space regex)
for sep in [',', '\t', r'\s+']:
    try:
        # Use parse_dates=['Date'] during initial read
        df_temp = pd.read_csv(file_path, sep=sep, engine='python', parse_dates=['Date'])
        # Check for expected column names (NOW USING THE CONFIG VARIABLE)
        if ASSET_PRICE_COLUMN_NAME in df_temp.columns and 'Inflation' in df_temp.columns and 'Date' in df_temp.columns:
            df = df_temp
            console.print(f"SUCCESS: File loaded using separator: '{sep}'")
            loaded = True
            break
    except Exception:
        continue

# Attempt 2: Fixed-Width File (FWF) - For cases where data is run together with no delimiters.
if not loaded:
    try:
        # Estimated widths based on user's undelimited sample:
        widths = [10, 12, 12]
        names = ['Date', ASSET_PRICE_COLUMN_NAME, 'Inflation'] # Use the config variable here too

        # Try FWF skipping the first row (assuming it's a mangled header line)
        df_temp = pd.read_fwf(file_path, widths=widths, names=names, parse_dates=['Date'], skiprows=[0])

        # Check if FWF parse was successful by looking at the first price value
        if not df_temp.empty and pd.notna(df_temp.iloc[0][ASSET_PRICE_COLUMN_NAME]):
            df = df_temp
            console.print("SUCCESS: File loaded using Fixed-Width Format (FWF) detection (Skipped Header).")
            loaded = True

    except Exception:
        pass

# Check for loading failure and print a helpful message
if not loaded or df is None:
    console.print(f"[bold red]FATAL ERROR: Could not load data from {file_path}. The data is severely malformed (no delimiters). Ensure the file is present and properly delimited (e.g., using commas or tabs) or check the data format.[/bold red]")
    # If loading failed, exit gracefully
    exit()


# --- WRAP THE REST OF THE EXECUTION IN A CHECK TO PREVENT AttributeError ---
if df is not None:
    console.print(f"Initial raw data loaded: {len(df)} rows.")
    df = df.set_index('Date')

    # FIX: Explicitly ensure the index is a DatetimeIndex to prevent the TypeError during filtering.
    # We use format='mixed' to handle the inconsistent date formats.
    try:
        df.index = pd.to_datetime(df.index, format='mixed', dayfirst=False)
        console.print("Index successfully converted to DatetimeIndex using mixed format inference.")
    except Exception as e:
        console.print(f"[bold red]CRITICAL WARNING: Could not convert index to datetime objects. Filtering may fail: {e}[/bold red]")

    # Use errors='coerce' to turn any remaining malformed strings into NaN for cleaning
    df[ASSET_PRICE_COLUMN_NAME] = pd.to_numeric(df[ASSET_PRICE_COLUMN_NAME], errors='coerce')
    df['Inflation'] = pd.to_numeric(df['Inflation'], errors='coerce')

    # 1. Drop rows with invalid data
    # Check for both the price column and 'Inflation'
    df_cleaned_initial = df.dropna(subset=[ASSET_PRICE_COLUMN_NAME, 'Inflation']).copy() # Use .copy() to ensure a clean DataFrame
    rows_dropped_initial = len(df) - len(df_cleaned_initial)
    df = df_cleaned_initial

    # Generalized output message for cleaning
    console.print(f"Data rows dropped due to malformed Investment/Inflation values: {rows_dropped_initial} rows. (Check raw file if this number is large)")
    console.print(f"Data size after initial cleaning: {len(df)} rows.")

    # --- FEATURE ENGINEERING FOR VOLATILITY & SKEWNESS REGIMES ---

    # We use .loc to prevent the SettingWithCopyWarning
    # 1. Daily Nominal Log Returns (%)
    # NOW USING THE CONFIG VARIABLE HERE
    df.loc[:, 'Daily_Nominal_Returns'] = np.log(df[ASSET_PRICE_COLUMN_NAME] / df[ASSET_PRICE_COLUMN_NAME].shift(1)) * 100

    # 2. Daily Inflation Log Change (%) (Assuming 'Inflation' is an index level like CPI)
    df.loc[:, 'Daily_Inflation_Log_Change'] = np.log(df['Inflation'] / df['Inflation'].shift(1)) * 100

    # 3. Daily Real Log Returns (%) (Nominal - Inflation change)
    df.loc[:, 'Daily_Real_Returns'] = df['Daily_Nominal_Returns'] - df['Daily_Inflation_Log_Change']

    # 4. HMM Feature 1: Monthly (21-day) Annualized Volatility of Real Returns
    df.loc[:, 'Volatility'] = df['Daily_Real_Returns'].rolling(VOLATILITY_WINDOW).std() * np.sqrt(TRADING_DAYS_YEAR)

    # 5. HMM Feature 2: Rolling 12-month (252-day) Skewness of Real Returns
    # Calculate rolling skewness using scipy.stats.skew
    from scipy.stats import skew
    df.loc[:, 'Skewness'] = df['Daily_Real_Returns'].rolling(SKEWNESS_WINDOW).apply(
        lambda x: skew(x.dropna()) if x.notna().sum() >= SKEWNESS_WINDOW * 0.8 else np.nan,
        raw=False
    )

    # Drop initial NA values caused by rolling windows (need both vol and skew to be valid)
    df_cleaned_rolling = df.dropna(subset=['Volatility', 'Skewness']).copy() # Use .copy() again
    rows_dropped_rolling = len(df) - len(df_cleaned_rolling)
    df = df_cleaned_rolling

    console.print(f"Data rows dropped due to rolling windows (vol: {VOLATILITY_WINDOW} days, skew: {SKEWNESS_WINDOW} days): {rows_dropped_rolling} rows.")
    console.print(f"Daily data size ready for HMM: {len(df)} rows.")

    # --- NEW FEATURE: FILTER BY SELECTED START DATE ---
    if START_DATE_ANALYSIS is not None:
        start_date = pd.to_datetime(START_DATE_ANALYSIS)
        df_filtered = df[df.index >= start_date].copy()

        rows_dropped_date = len(df) - len(df_filtered)
        df = df_filtered

        console.print(f"Data rows dropped due to filtering before {START_DATE_ANALYSIS}: {rows_dropped_date} rows.")
        console.print(f"Final daily data size after date filter: {len(df)} rows.")

    if DEBUG_MODE and not df.empty:
        console.print("\n[bold yellow]DEBUG:[/bold yellow] Features DataFrame Head (After Lookback & Date Drop):")
        console.print(df[['Daily_Real_Returns', 'Volatility', 'Skewness']].head())

    # --- RESAMPLE TO NON-OVERLAPPING MONTHLY DATA (21-Day Step) ---
    # Since LOOKBACK_PERIOD=21, this creates non-overlapping monthly steps.
    df_monthly = df.iloc[::LOOKBACK_PERIOD].copy()

    if DEBUG_MODE:
        console.print(f"\n[bold yellow]DEBUG:[/bold yellow] Monthly Resampling applied (iloc[::{LOOKBACK_PERIOD}]). Total monthly samples: {len(df_monthly)}.")
        console.print("\n[bold yellow]DEBUG:[/bold yellow] Monthly Resampled Data Head (Data used for HMM training):")
        console.print(df_monthly[['Volatility', 'Skewness']].head())

    # Print the analysis period
    if not df_monthly.empty:
        # Index is now guaranteed to be DatetimeIndex
        start_date_actual = df_monthly.index.min().strftime('%Y-%m-%d')
        end_date_actual = df_monthly.index.max().strftime('%Y-%m-%d')
        console.print(f"ANALYSIS PERIOD: {start_date_actual} to {end_date_actual} (Based on non-overlapping {LOOKBACK_PERIOD}-day data points).", style="bold green")

    # Final features used for HMM: Volatility and Skewness (2D feature space)
    X_data = df_monthly[['Volatility', 'Skewness']].values
    console.print(f"HMM Input Features Shape: {X_data.shape}.")
    console.print(f"  Features: Non-Overlapping Monthly Annualized Volatility + 12-Month Rolling Skewness")

    if X_data.shape[0] < NUM_REGIMES:
        console.print(f"[bold red]ERROR: Not enough non-overlapping monthly steps ({X_data.shape[0]}) to fit the HMM with {NUM_REGIMES} regimes. Adjust START_DATE_ANALYSIS or increase data length.[/bold red]")
        # Since df is not None, we can stop here if data is insufficient.
    else:
        # --- 3. HMM INFERENCE ---
        console.print("\n--- Running HMM to Infer Regimes (Volatility & Skewness) ---", style="bold yellow")

        # Covariance type 'diag' for 2D feature space (volatility and skewness)
        model_inferred = hmm.GaussianHMM(n_components=NUM_REGIMES, covariance_type="diag", n_iter=2000, tol=1e-5, random_state=RANDOM_SEED)

        try:
            model_inferred.fit(X_data)
            if DEBUG_MODE:
                console.print("\n[bold yellow]DEBUG:[/bold yellow] HMM fit successful after 2000 iterations (or convergence).")
        except ValueError as e:
            console.print(f"[bold red]Error during HMM fit: {e}. HMM can be sensitive to data distribution.[/bold red]")
            # If HMM fit fails, stop execution
            exit()

        df_monthly['Original_Inferred_Regime'] = model_inferred.predict(X_data)
        inferred_transmat = model_inferred.transmat_

        if DEBUG_MODE:
            console.print("\n[bold yellow]DEBUG:[/bold yellow] HMM Inferred Means and Covariances:")
            # For diag covariance with 2D feature, covars_ is shape (n_components, n_features, n_features)
            # which is actually a 2x2 diagonal matrix for each regime. Extract diagonal elements.
            for i in range(NUM_REGIMES):
                mean_vol = float(model_inferred.means_[i, 0])
                mean_skew = float(model_inferred.means_[i, 1])
                # Extract diagonal elements: [i, 0, 0] for vol variance, [i, 1, 1] for skew variance
                vol_variance = float(model_inferred.covars_[i, 0, 0])
                skew_variance = float(model_inferred.covars_[i, 1, 1])
                console.print(f"  Regime {i}: Mean Volatility = {mean_vol:.4f}, Mean Skewness = {mean_skew:.4f}")
                console.print(f"            Vol Variance = {vol_variance:.4f}, Skew Variance = {skew_variance:.4f}")

        # --- DYNAMICALLY MAP HMM INDICES (0-2) BASED ON VOLATILITY AND SKEWNESS ---

        # 1. Collect HMM state means (both volatility and skewness)
        regime_data = []
        for i in range(NUM_REGIMES):
            regime_data.append({
                'original_index': i,
                'mean_vol': model_inferred.means_[i, 0],
                'mean_skew': model_inferred.means_[i, 1],
            })

        # 2. Sort regimes by volatility first (primary), then skewness (secondary)
        # This helps create interpretable regime labels
        regime_data.sort(key=lambda x: (x['mean_vol'], x['mean_skew']), reverse=False)
        
        if DEBUG_MODE:
            console.print("\n[bold yellow]DEBUG:[/bold yellow] Regime sorting (by volatility, then skewness):")
            for i, data in enumerate(regime_data):
                console.print(f"  Sorted position {i}: Original index {data['original_index']}, "
                            f"Mean Vol={data['mean_vol']:.2f}, Mean Skew={data['mean_skew']:.2f}")

        # 3. Assign labels based on vol and skew characteristics
        REGIME_LABELS = {}
        REGIME_INDEX_MAP = {} # {Original HMM Index: New Index (0, 1, 2)}
        sorted_regime_presentation_data = [] # For presentation tables

        for i, data in enumerate(regime_data):
            original_hmm_index = data['original_index']
            final_index = i  # Simple sequential mapping
            
            # Create descriptive label based on vol and skew
            vol_level = "Low" if data['mean_vol'] < 15 else ("Med" if data['mean_vol'] < 25 else "High")
            skew_level = "Neg" if data['mean_skew'] < -0.3 else ("Neut" if data['mean_skew'] < 0.3 else "Pos")
            regime_label = f"R{i+1}: {vol_level} Vol, {skew_level} Skew"

            # Map 1: Original HMM Index -> New Index (0, 1, 2)
            REGIME_INDEX_MAP[original_hmm_index] = final_index

            # Map 2: Original HMM ID (R1, R2, R3) -> Canonical Label
            regime_id = f'R{original_hmm_index+1}'
            REGIME_LABELS[regime_id] = regime_label

            # Prepare data for presentation tables
            sorted_regime_presentation_data.append({
                'HMM ID': regime_id,
                'Label': regime_label,
                'original_index': original_hmm_index,
                'final_index': final_index,
                'mean_vol': data['mean_vol'],
                'mean_skew': data['mean_skew'],
            })

        # 4. Apply the new index mapping to the monthly data
        df_monthly.loc[:, 'Inferred_Regime_Ranked'] = df_monthly['Original_Inferred_Regime'].map(REGIME_INDEX_MAP)

        if DEBUG_MODE:
            console.print("\n[bold yellow]DEBUG:[/bold yellow] Final Regime Index Map:")
            console.print(REGIME_INDEX_MAP)
            console.print("\n[bold yellow]DEBUG:[/bold yellow] Monthly Data Head with New Index:")
            console.print(df_monthly[['Original_Inferred_Regime', 'Inferred_Regime_Ranked']].head())

        # --- MAP MONTHLY REGIME BACK TO DAILY INDEX FOR RISK CALCULATION AND VISUALIZATION ---
        regime_map = df_monthly['Inferred_Regime_Ranked'].to_dict()
        # Forward-fill the monthly regimes to the daily index for risk calculations and visualization context
        # Use .loc to avoid SettingWithCopyWarning
        temp_series = pd.Series(df.index.map(regime_map), index=df.index).ffill()
        df.loc[:, 'Visual_Regime'] = temp_series

        # --- 4. CALCULATE REGIME DURATION METRICS ---

        # A. Inferred Transition Matrix (P) - SORTED BY VOLATILITY AND SKEWNESS
        console.print(f"\n--- A. Inferred Transition Probability Matrix (P) (Monthly/21-Day Step) - Sorted by Volatility & Skewness ---", style="bold blue")

        # Get the original HMM indices in the sorted order
        sorted_indices = [data['original_index'] for data in sorted_regime_presentation_data]
        sorted_labels = [f"{data['final_index']}: {data['Label']}" for data in sorted_regime_presentation_data] # Label with the new 0, 1, 2 index

        # Re-index the transition matrix using the skewness-sorted indices
        transmat_sorted = inferred_transmat[np.ix_(sorted_indices, sorted_indices)]

        df_transmat = pd.DataFrame(transmat_sorted, columns=sorted_labels, index=sorted_labels)

        table_a = Table(title="HMM Transition Matrix (P) [Rows: From, Cols: To]", show_header=True, header_style="bold cyan")
        table_a.add_column("From / To", style="dim", justify="left")
        for col in df_transmat.columns:
            table_a.add_column(col, justify="right")

        for index, row in df_transmat.iterrows():
            table_a.add_row(
                index,
                *[f"{val:.4f}" for val in row.values]
            )
        console.print(table_a)
        
        # Save transition matrix to CSV for use in simulations
        import os
        os.makedirs('data', exist_ok=True)
        transmat_output_file = "data/HMM_transition_matrix.csv"
        df_transmat.to_csv(transmat_output_file)
        console.print(f"[green]✓ Saved HMM transition matrix to {transmat_output_file}[/green]")


        # B. Calculate Mean Duration and Std Dev of Duration for each regime - SORTED
        duration_metrics = []
        for data in sorted_regime_presentation_data:
            original_i = data['original_index']
            final_index = data['final_index']

            # Get P_ii from the SORTED transition matrix (which may have been adjusted)
            # Map original index to sorted position
            sorted_pos = sorted_indices.index(original_i)
            p_ii = transmat_sorted[sorted_pos, sorted_pos]

            mean_duration_steps = 1 / (1 - p_ii)
            std_dev_duration_steps = np.sqrt(p_ii) / (1 - p_ii)
            # Since a step is 21 trading days (approx 1 month), divide by 12 for years.
            ann_mean_duration = mean_duration_steps / 12

            duration_metrics.append({
                'Index': final_index,
                'Label': data['Label'],
                'P(Stay in R_i)': p_ii,
                'Mean Duration (Steps/Months)': mean_duration_steps,
                'Mean Duration (Years)': ann_mean_duration,
                'Std Dev Duration (Steps/Months)': std_dev_duration_steps
            })

        df_metrics = pd.DataFrame(duration_metrics)
        console.print(f"\n--- B. Regime Duration Statistics (Endogenous Window - Monthly/21-Day) - Sorted by Volatility & Skewness ---", style="bold blue")

        table_b = Table(title="Regime Duration Metrics", show_header=True, header_style="bold cyan")
        # Adjusting column headers to reflect steps/months
        table_b.add_column('Index', justify="center")
        table_b.add_column('Label', justify="center")
        table_b.add_column('P(Stay in R_i)', justify="right")
        table_b.add_column('Mean Duration (Steps/Months)', justify="right")
        table_b.add_column('Mean Duration (Years)', justify="right")
        table_b.add_column('Std Dev Duration (Steps/Months)', justify="right")

        for index, row in df_metrics.iterrows():
            table_b.add_row(
                f"{int(row['Index'])}",
                row['Label'],
                f"{row['P(Stay in R_i)']:.3f}",
                f"{row['Mean Duration (Steps/Months)']:.3f}",
                f"{row['Mean Duration (Years)']:.3f}",
                f"{row['Std Dev Duration (Steps/Months)']:.3f}"
            )
        console.print(table_b)


        # C. Long-run (Stationary) Probability - SORTED
        # Calculate stationary probabilities using the SORTED transition matrix (which may have been adjusted)
        eigenvalues, eigenvectors = np.linalg.eig(transmat_sorted.T)
        stationary_probs = eigenvectors[:, np.isclose(eigenvalues, 1)][:, 0]
        stationary_probs = stationary_probs / np.sum(stationary_probs)
        stationary_probs = np.real(stationary_probs)

        console.print(f"\n--- C. Long-Run (Stationary) Probability of Being in Regime (Unconditional) - Sorted by Volatility & Skewness ---", style="bold blue")
        stationary_data = []

        # Map sorted stationary probabilities to the sorted presentation data
        for i, data in enumerate(sorted_regime_presentation_data):
            stationary_data.append({
                'Index': data['final_index'],
                'Regime Label': data['Label'],
                'Stationary Probability': stationary_probs[i]  # Use position i in sorted order
            })

        df_stationary = pd.DataFrame(stationary_data)

        table_c = Table(title="Stationary Probability", show_header=True, header_style="bold cyan")
        for col in df_stationary.columns:
            table_c.add_column(col, justify="center" if col in ['Index', 'Regime Label'] else "right")

        for index, row in df_stationary.iterrows():
            table_c.add_row(
                f"{int(row['Index'])}",
                row['Regime Label'],
                f"{row['Stationary Probability']:.4f}"
            )
        console.print(table_c)


        # D. Inferred Regime Characteristics (Means and Risk Metrics) - Sorted by Volatility & Skewness
        console.print("\n--- D. Inferred Regime Characteristics (Means and Risk Metrics) - Sorted by Volatility & Skewness ---", style="bold blue")

        means_data = []
        for data in sorted_regime_presentation_data:
            i = data['original_index'] # Use original HMM index to filter daily data

            mean_vol = data['mean_vol']
            mean_skew = data['mean_skew']
            unique_label = data['Label']
            final_index = data['final_index']

            # --- Calculate Derived Metrics (Mean Return & Other Risk Metrics) using daily data ---
            # NOTE: Must filter the daily data using the NEW 'Visual_Regime' column which holds the 0, 1, 2 index.
            regime_daily_data = df[df['Visual_Regime'] == final_index]
            regime_daily_returns = regime_daily_data['Daily_Nominal_Returns'].dropna() # Nominal Returns
            regime_daily_real_returns = regime_daily_data['Daily_Real_Returns'].dropna() # Real Returns (core feature)

            if not regime_daily_returns.empty:
                # Annual Nominal Return
                annual_nominal_return = regime_daily_returns.mean() * TRADING_DAYS_YEAR
                # Annual Real Return (derived)
                annual_real_return_derived = regime_daily_real_returns.mean() * TRADING_DAYS_YEAR
                
                # Calculate annualized volatility for this regime (derived from actual returns)
                annual_volatility_derived = regime_daily_real_returns.std() * np.sqrt(TRADING_DAYS_YEAR)

                # Calculate Max Drawdown
                max_drawdown = calculate_max_drawdown(regime_daily_returns)
            else:
                annual_nominal_return = 0.0
                annual_real_return_derived = 0.0
                annual_volatility_derived = 0.0
                max_drawdown = 0.0
            # --- END Derived Metrics Calculation ---

            means_data.append({
                'Index': final_index,
                'Regime Label': unique_label,
                'FEATURE: Volatility [HMM MEAN]': mean_vol, # HMM-inferred mean - This is the DEFINITIVE volatility
                'FEATURE: Skewness [HMM MEAN]': mean_skew, # HMM-inferred mean - This is the DEFINITIVE skewness
                'Annual Volatility (%) (Derived)': annual_volatility_derived,
                'Annual Real Return (%) (Derived)': annual_real_return_derived,
                'Annual Nominal Return (%) (Derived)': annual_nominal_return,
                'Max Drawdown (%)': max_drawdown
            })

        df_means = pd.DataFrame(means_data)

        # Build Rich Table D - FINALIZED COLUMN HEADINGS
        table_d = Table(title="Inferred Regime Characteristics (Annualized)", show_header=True, header_style="bold cyan")
        table_d.add_column('Index', justify='center')
        table_d.add_column('Regime Label', justify='left') # Adjusted for longer labels
        # These are the HMM feature means, which are the true definitions of the regime's characteristics.
        table_d.add_column('FEATURE: Volatility [HMM MEAN]', justify='right', style='bold yellow')
        table_d.add_column('FEATURE: Skewness [HMM MEAN]', justify='right', style='bold yellow')
        table_d.add_column('Annual Volatility (%) (Derived)', justify='right', style='dim')
        table_d.add_column('Annual Real Return (%) (Derived)', justify='right', style='dim')
        table_d.add_column('Annual Nominal Return (%) (Derived)', justify='right', style='dim')
        table_d.add_column('Max Drawdown (%)', justify='right', style='bold red')

        for index, row in df_means.iterrows():
            table_d.add_row(
                f"{int(row['Index'])}",
                row['Regime Label'],
                f"{row['FEATURE: Volatility [HMM MEAN]']:.2f}",
                f"{row['FEATURE: Skewness [HMM MEAN]']:.4f}",
                f"{row['Annual Volatility (%) (Derived)']:.2f}",
                f"{row['Annual Real Return (%) (Derived)']:.2f}",
                f"{row['Annual Nominal Return (%) (Derived)']:.2f}",
                f"{row['Max Drawdown (%)']:.2f}"
            )
        console.print(table_d)

        # --- 4.5 EXPORT CLASSIFICATION TO CSV (Nominal Returns + Regime ID) ---
        export_data = []

        for idx, row in df_monthly.iterrows(): # Iterate over the monthly (21-day step) data points
            # Determine the slice of daily data for this monthly step
            start_loc = df.index.get_loc(idx)
            end_loc = start_loc + LOOKBACK_PERIOD
            if end_loc <= len(df):
                daily_slice = df.iloc[start_loc:end_loc]
            else:
                daily_slice = df.iloc[start_loc:]

            # Calculate the total nominal return for the 21-day period by summing the daily log returns.
            # This represents the total nominal percent change (monthly nominal return) for the period.
            monthly_nominal_return = daily_slice['Daily_Nominal_Returns'].sum()

            export_data.append({
                'Monthly Start Date': idx.strftime('%Y-%m-%d'), # Relabeling since LOOKBACK_PERIOD=21 (monthly step)
                'Total Nominal Return (%)': monthly_nominal_return,
                'Inferred Regime ID': row['Inferred_Regime_Ranked']
            })

        df_export = pd.DataFrame(export_data)
        import os
        os.makedirs('data', exist_ok=True)
        output_file_name = 'data/regime_classification_nominal_returns.csv'
        df_export.to_csv(output_file_name, index=False)

        console.print(f"SUCCESS: Monthly Nominal Return + Regime ID exported to {output_file_name}")

        # --- 5. VISUALIZATION (optional, controlled by ENABLE_PLOT flag) ---
        if ENABLE_PLOT:
            plt.style.use('dark_background')
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True)

            # Define colors for the regimes (expanded to support up to 6 regimes)
            colors = [
                '#1f77b4',  # 0=blue
                '#ff7f0e',  # 1=orange
                '#2ca02c',  # 2=green
                '#d62728',  # 3=red
                '#9467bd',  # 4=purple
                '#8c564b'   # 5=brown
            ]

            # Helper function to shade regimes on both plots
            def shade_regimes(ax):
                for i in range(len(df) - 1):
                    regime = df['Visual_Regime'].iloc[i]
                    if pd.notna(regime):
                        regime_index = int(regime)
                        if 0 <= regime_index < len(colors):
                            ax.axvspan(df.index[i], df.index[i+1], facecolor=colors[regime_index], alpha=0.2, linewidth=0)

            # --- PLOT 1: Volatility and Skewness ---
            ax1_twin = ax1.twinx()
            df['Volatility'].plot(ax=ax1, color='cyan', alpha=0.8, linewidth=1.5, label='Annualized Volatility (%)')
            df['Skewness'].plot(ax=ax1_twin, color='magenta', alpha=0.8, linewidth=1.5, label='12-Month Rolling Skewness')
            ax1.set_ylabel('Annualized Volatility (%)', color='cyan', fontsize=12)
            ax1_twin.set_ylabel('12-Month Rolling Skewness', color='magenta', fontsize=12)
            ax1.tick_params(axis='y', labelcolor='cyan')
            ax1_twin.tick_params(axis='y', labelcolor='magenta')
            ax1.grid(True, alpha=0.3)
            ax1_twin.axhline(y=0, color='magenta', linestyle='--', alpha=0.5, linewidth=1)  # Zero skewness line
            shade_regimes(ax1)
            # Combine legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax1_twin.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', facecolor='black', edgecolor='white', framealpha=0.8)
            ax1.set_title('Volatility & Skewness Over Time with Inferred Regimes', color='white', fontsize=14, fontweight='bold')

            # --- PLOT 2: Price (Log Scale) ---
            # Use the actual price data from the CSV
            df[ASSET_PRICE_COLUMN_NAME].plot(ax=ax2, color='white', alpha=0.9, linewidth=1.5, label=f'{ASSET_PRICE_COLUMN_NAME} Price')
            ax2.set_yscale('log')  # Set y-axis to log scale
            ax2.set_ylabel('Price (Log Scale)', color='white', fontsize=12)
            ax2.set_xlabel('Date', color='white', fontsize=12)
            ax2.tick_params(axis='y', labelcolor='white')
            ax2.grid(True, alpha=0.3, which='both')  # Show both major and minor grid lines for log scale
            shade_regimes(ax2)
            ax2.set_title('Asset Price Over Time with Inferred Regimes (Log Scale)', color='white', fontsize=14, fontweight='bold')

            # Add regime legend to the price plot
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=colors[data['final_index']], edgecolor='none', alpha=0.5,
                                     label=f"{data['final_index']}: {data['Label']}")
                            for data in sorted_regime_presentation_data]
            ax2.legend(handles=legend_elements, loc='upper left', title='Inferred Vol & Skew Regimes',
                      facecolor='black', edgecolor='white', framealpha=0.8)

            # Add overall title with asset name
            asset_name_safe = ASSET_PRICE_COLUMN_NAME.replace('/', '_').replace('\\', '_').replace(':', '_')
            fig.suptitle(f'HMM Volatility & Skewness Regime Detection - {ASSET_PRICE_COLUMN_NAME} - {NUM_REGIMES} Regimes', color='white', fontsize=16, fontweight='bold', y=0.995)
            
            plt.tight_layout()
            
            # Save plot to output folder
            import os
            os.makedirs('output', exist_ok=True)
            plot_filename = f'output/HMM_VolSkew_Regimes_{asset_name_safe}_{NUM_REGIMES}Regimes.png'
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight', facecolor='black')
            console.print(f"[green]✓ Plot saved to {plot_filename}[/green]")
            
            plt.show()
        else:
            console.print("[dim]Plotting disabled (ENABLE_PLOT = False)[/dim]")