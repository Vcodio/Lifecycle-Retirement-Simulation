import numpy as np
import pandas as pd
from hmmlearn import hmm
import matplotlib.pyplot as plt
import math
from scipy.stats import norm
from rich.console import Console
from rich.table import Table

# --- 1. CONFIGURATION ---
NUM_REGIMES = 3
LOOKBACK_PERIOD = 21
RANDOM_SEED = 42
DEBUG_MODE = True 

# Set to None to use all available data after lookback/cleaning. Format: 'YYYY-MM-DD'
START_DATE_ANALYSIS = '1800-01-01' 

# If you change your CSV column name, update this variable.
ASSET_PRICE_COLUMN_NAME = 'S&P 500' 


np.random.seed(RANDOM_SEED)
TRADING_DAYS_YEAR = 252 # Used for annualizing volatility and returns
ANNUALIZATION_FACTOR = 4 # 4 quarters per year

# Naming the regime labels. These will be dynamically assigned based on the HMM mean volatility.
CANONICAL_VOL_RANKING = [
    'V-1: Lowest Volatility',
    'V-2: Medium Volatility',
    'V-3: Highest Volatility',
]
# NEW: Defines the desired output index for the final classification (0=Low, 1=Medium, 2=High)
DESIRED_VOL_INDEX = {
    'V-1: Lowest Volatility': 0,
    'V-2: Medium Volatility': 1,
    'V-3: Highest Volatility': 2,
}

REGIME_LABELS = {} # This will store the final HMM Index -> Canonical Label mapping
VOL_RANKED_INDEX_MAP = {} # This will store the original HMM index -> desired 0, 1, 2 index map

# Initialize Rich Console for all output
console = Console()

# Max Drawdown Calc!
def calculate_max_drawdown(daily_log_returns_series):
    """Calculates Max Drawdown from a series of daily log returns (in percent)."""
    if daily_log_returns_series.empty:
        return 0.0
    # Convert log returns (%) to simple returns (ratio)
    # The log return is in percent, so divide by 100 before taking exp/subtracting 1
    simple_returns = np.exp(daily_log_returns_series / 100) - 1
    # Calculate cumulative simple returns
    cumulative_returns = (1 + simple_returns).cumprod()
    # Calculate running maximum
    running_max = cumulative_returns.cummax()
    # Calculate drawdown
    drawdown = (cumulative_returns / running_max) - 1
    # Max drawdown is the minimum value (most negative)
    return drawdown.min() * 100

# --- 2. DATA LOADING & PREPROCESSING ---
console.print("\n--- 2. Loading and Preprocessing Historical Data ---", style="bold")
file_path = "Market and Inflation - Regime Testing.csv"
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


# --- WRAP THE REST OF THE EXECUTION IN A CHECK TO PREVENT AttributeError ---
if df is not None:
    console.print(f"Initial raw data loaded: {len(df)} rows.")
    df = df.set_index('Date')
    
    # FIX: Making sure the index is a DatetimeIndex to prevent the TypeError during filtering.
    # Using format='mixed' to handle possible inconsistent date formats.
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

    # --- FEATURE ENGINEERING FOR VOLATILITY REGIMES ---

    # We use .loc to prevent the SettingWithCopyWarning
    # 1. Daily Nominal Log Returns (%)
    # NOW USING THE CONFIG VARIABLE HERE
    df.loc[:, 'Daily_Nominal_Returns'] = np.log(df[ASSET_PRICE_COLUMN_NAME] / df[ASSET_PRICE_COLUMN_NAME].shift(1)) * 100

    # 2. Daily Inflation Log Change (%) (Assuming 'Inflation' is an index level like CPI)
    df.loc[:, 'Daily_Inflation_Log_Change'] = np.log(df['Inflation'] / df['Inflation'].shift(1)) * 100

    # 3. Daily Real Log Returns (%) (Nominal - Inflation change)
    df.loc[:, 'Daily_Real_Returns'] = df['Daily_Nominal_Returns'] - df['Daily_Inflation_Log_Change']

    # 4. HMM Feature: Annualized Volatility of Real Returns 
    # Calculate rolling standard deviation (daily vol) and annualize it.
    df.loc[:, 'Volatility'] = df['Daily_Real_Returns'].rolling(LOOKBACK_PERIOD).std() * np.sqrt(TRADING_DAYS_YEAR)

    # 2. Drop initial NA values caused by rolling window
    df_cleaned_rolling = df.dropna(subset=['Volatility']).copy() # Use .copy() again
    rows_dropped_rolling = len(df) - len(df_cleaned_rolling)
    df = df_cleaned_rolling

    console.print(f"Data rows dropped due to {LOOKBACK_PERIOD}-day rolling lookback (standard loss): {rows_dropped_rolling} rows.")
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
        console.print(df[['Daily_Real_Returns', 'Volatility']].head())

    
    # --- RESAMPLE TO NON-OVERLAPPING DATA ---
    df_quarterly = df.iloc[::LOOKBACK_PERIOD].copy()

    if DEBUG_MODE:
        console.print(f"\n[bold yellow]DEBUG:[/bold yellow] Quarterly Resampling applied (iloc[::{LOOKBACK_PERIOD}]). Total quarterly samples: {len(df_quarterly)}.")
        console.print("\n[bold yellow]DEBUG:[/bold yellow] Quarterly Resampled Data Head (Data used for HMM training):")
        console.print(df_quarterly[['Volatility']].head())

    # Print the analysis period
    if not df_quarterly.empty:
        # Index is now guaranteed to be DatetimeIndex
        start_date_actual = df_quarterly.index.min().strftime('%Y-%m-%d')
        end_date_actual = df_quarterly.index.max().strftime('%Y-%m-%d')
        console.print(f"ANALYSIS PERIOD: {start_date_actual} to {end_date_actual} (Based on non-overlapping quarterly data points).", style="bold green")

    # Final feature used for HMM: Volatility. Must be reshaped to (n_samples, n_features)
    X_data = df_quarterly['Volatility'].values.reshape(-1, 1)
    # --- CHANGE 1: Remove "ONLY" ---
    console.print(f"HMM Input Features Shape: {X_data.shape}. Used feature: Non-Overlapping Quarterly Annualized Volatility.")

    if X_data.shape[0] < NUM_REGIMES:
        console.print(f"[bold red]ERROR: Not enough non-overlapping quarters ({X_data.shape[0]}) to fit the HMM with {NUM_REGIMES} regimes. Adjust START_DATE_ANALYSIS or increase data length.[/bold red]")
        # Since df is not None, we can stop here if data is insufficient.
    else:
        # --- 3. HMM INFERENCE ---

        
        console.print("\n--- Running HMM to Infer Regimes (Volatility) ---", style="bold yellow")

        # Covariance type can be 'full' or 'diag' for 1 feature, as it's just a scalar variance.
        model_inferred = hmm.GaussianHMM(n_components=NUM_REGIMES, covariance_type="full", n_iter=2000, tol=1e-5, random_state=RANDOM_SEED)

        try:
            model_inferred.fit(X_data)
            if DEBUG_MODE:
                console.print("\n[bold yellow]DEBUG:[/bold yellow] HMM fit successful after 2000 iterations (or convergence).")
        except ValueError as e:
            console.print(f"[bold red]Error during HMM fit: {e}. HMM can be sensitive to data distribution.[/bold red]")
            # If HMM fit fails, stop execution
            exit()

        df_quarterly['Original_Inferred_Regime'] = model_inferred.predict(X_data)
        inferred_transmat = model_inferred.transmat_
        
        if DEBUG_MODE:
            console.print("\n[bold yellow]DEBUG:[/bold yellow] HMM Inferred Means and Covariances (Features are Volatility):")
            for i in range(NUM_REGIMES):
                console.print(f"  Regime {i}: Mean Volatility = {model_inferred.means_[i, 0]:.4f}, Variance = {model_inferred.covars_[i, 0, 0]:.4f}")


        # --- DYNAMICALLY MAP HMM INDICES (0-2) TO VOLATILITY RANKING ---

        # 1. Collect HMM state means
        regime_data = []
        for i in range(NUM_REGIMES):
            regime_data.append({
                'original_index': i,
                # Volatility is Feature 0
                'mean_vol': model_inferred.means_[i, 0],
            })

        # 2. Sort the regimes by their mean volatility (Feature 0) in ascending order (Lowest to Highest)
        regime_data.sort(key=lambda x: x['mean_vol'], reverse=False)
        
        # 3. Assign the new volatility-ranked labels AND the new 0, 1, 2 index
        REGIME_LABELS = {}
        VOL_RANKED_INDEX_MAP = {} # {Original HMM Index: New Volatility Index (0, 1, or 2)}
        sorted_regime_presentation_data = [] # For presentation tables

        for i, data in enumerate(regime_data):
            # i = 0 -> V-1: Lowest Volatility, i = 1 -> V-2: Medium Volatility, i = 2 -> V-3: Highest Volatility
            vol_rank_label = CANONICAL_VOL_RANKING[i] 
            final_vol_index = DESIRED_VOL_INDEX[vol_rank_label] # This is the desired 0, 1, or 2
            original_hmm_index = data['original_index']
            
            # Map 1: Original HMM Index -> New Volatility Index (0, 1, or 2)
            VOL_RANKED_INDEX_MAP[original_hmm_index] = final_vol_index
            
            # Map 2: Original HMM ID (R1, R2, R3) -> Canonical Label
            regime_id = f'R{original_hmm_index+1}'
            REGIME_LABELS[regime_id] = vol_rank_label
            
            # Prepare data for presentation tables (already sorted V-1, V-2, V-3)
            sorted_regime_presentation_data.append({
                'HMM ID': regime_id,
                'Label': vol_rank_label,
                'original_index': original_hmm_index,
                'final_index': final_vol_index, # The new 0, 1, 2 index
                'mean_vol': data['mean_vol'],
            })
            
        # 4. Apply the new index mapping to the quarterly data
        df_quarterly.loc[:, 'Inferred_Regime_VolRanked'] = df_quarterly['Original_Inferred_Regime'].map(VOL_RANKED_INDEX_MAP)

        if DEBUG_MODE:
            console.print("\n[bold yellow]DEBUG:[/bold yellow] Final Volatility-Ranked Index Map:")
            console.print(VOL_RANKED_INDEX_MAP)
            console.print("\n[bold yellow]DEBUG:[/bold yellow] Quarterly Data Head with New Index:")
            console.print(df_quarterly[['Original_Inferred_Regime', 'Inferred_Regime_VolRanked']].head())

        # --- MAP QUARTERLY REGIME BACK TO DAILY INDEX FOR RISK CALCULATION AND VISUALIZATION ---
        regime_map = df_quarterly['Inferred_Regime_VolRanked'].to_dict()
        # Forward-fill the quarterly regimes to the daily index for risk calculations and visualization context
        # Use .loc to avoid SettingWithCopyWarning
        temp_series = pd.Series(df.index.map(regime_map), index=df.index).ffill()
        df.loc[:, 'Visual_Regime'] = temp_series 


        # --- 4. CALCULATE REGIME DURATION METRICS ---

        # A. Inferred Transition Matrix (P) - SORTED BY VOLATILITY RANK (0, 1, 2)
        console.print("\n--- A. Inferred Transition Probability Matrix (P) (Quarterly Step) - Sorted by Volatility Rank (0, 1, 2) ---", style="bold blue")
        
        # Get the original HMM indices in the volatility-sorted order
        sorted_indices = [data['original_index'] for data in sorted_regime_presentation_data]
        sorted_labels = [f"{data['final_index']}: {data['Label']}" for data in sorted_regime_presentation_data] # Label with the new 0, 1, 2 index
        
        # Re-index the transition matrix using the volatility-sorted indices
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


        # B. Calculate Mean Duration and Std Dev of Duration for each regime - SORTED
        duration_metrics = []
        for data in sorted_regime_presentation_data:
            i = data['original_index']
            final_index = data['final_index']
            
            # Get P_ii from the original (unsorted) transition matrix using the original index i
            p_ii = inferred_transmat[i, i] 
            
            mean_duration_quarters = 1 / (1 - p_ii)
            std_dev_duration_quarters = np.sqrt(p_ii) / (1 - p_ii)
            ann_mean_duration = mean_duration_quarters / 4

            duration_metrics.append({
                'Index': final_index,
                'Label': data['Label'],
                'P(Stay in R_i)': p_ii,
                'Mean Duration (Qtrs)': mean_duration_quarters,
                'Mean Duration (Years)': ann_mean_duration,
                'Std Dev Duration (Qtrs)': std_dev_duration_quarters
            })

        df_metrics = pd.DataFrame(duration_metrics)
        console.print("\n--- B. Regime Duration Statistics (Endogenous Window - Quarterly) - Sorted by Volatility Rank (0, 1, 2) ---", style="bold blue")

        table_b = Table(title="Regime Duration Metrics", show_header=True, header_style="bold cyan")
        for col in df_metrics.columns:
            table_b.add_column(col, justify="center" if col in ['Index', 'Label'] else "right")

        for index, row in df_metrics.iterrows():
            table_b.add_row(
                f"{int(row['Index'])}",
                row['Label'],
                f"{row['P(Stay in R_i)']:.3f}",
                f"{row['Mean Duration (Qtrs)']:.3f}",
                f"{row['Mean Duration (Years)']:.3f}",
                f"{row['Std Dev Duration (Qtrs)']:.3f}"
            )
        console.print(table_b)


        # C. Long-run (Stationary) Probability - SORTED
        # Calculate stationary probabilities using the original (unsorted) transition matrix
        eigenvalues, eigenvectors = np.linalg.eig(inferred_transmat.T)
        stationary_probs = eigenvectors[:, np.isclose(eigenvalues, 1)][:, 0]
        stationary_probs = stationary_probs / np.sum(stationary_probs)
        stationary_probs = np.real(stationary_probs)

        console.print("\n--- C. Long-Run (Stationary) Probability of Being in Regime (Unconditional) - Sorted by Volatility Rank (0, 1, 2) ---", style="bold blue")
        stationary_data = []
        
        # Map original stationary probabilities to the sorted presentation data
        for data in sorted_regime_presentation_data:
            i = data['original_index']
            stationary_data.append({
                'Index': data['final_index'],
                'Regime Label': data['Label'],
                'Stationary Probability': stationary_probs[i]
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


        # D. Inferred Regime Characteristics (Means and Risk Metrics) - Sorted by Volatility Rank (0, 1, 2) ---
        console.print("\n--- D. Inferred Regime Characteristics (Means and Risk Metrics) - Sorted by Volatility Rank (0, 1, 2) ---", style="bold blue")
        
        means_data = []
        for data in sorted_regime_presentation_data:
            i = data['original_index'] # Use original HMM index to filter daily data
            
            mean_vol_annual = data['mean_vol']
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
                
                # We no longer calculate or report the derived daily volatility to avoid the inversion confusion.
                
                # Calculate Max Drawdown
                max_drawdown = calculate_max_drawdown(regime_daily_returns)
            else:
                annual_nominal_return = 0.0
                annual_real_return_derived = 0.0
                max_drawdown = 0.0
            # --- END Derived Metrics Calculation ---

            means_data.append({
                'Index': final_index,
                'Regime Label': unique_label,
                'FEATURE: ANNUAL Volatility (%) (Real)': mean_vol_annual, # HMM-inferred mean - This is the DEFINITIVE volatility
                'Annual Real Return (%) (Derived)': annual_real_return_derived,
                'Annual Nominal Return (%) (Derived)': annual_nominal_return,
                'Max Drawdown (%)': max_drawdown
            })

        df_means = pd.DataFrame(means_data)

        # Build Rich Table D - FINALIZED COLUMN HEADINGS
        table_d = Table(title="Inferred Regime Characteristics (Annualized)", show_header=True, header_style="bold cyan")
        table_d.add_column('Index', justify='center')
        table_d.add_column('Regime Label', justify='left') # Adjusted for longer labels
        # This is the HMM feature mean, which is the true definition of the regime's volatility level.
        table_d.add_column('FEATURE: ANNUAL Volatility (%) (Real) [HMM MEAN]', justify='right', style='bold yellow') 
        table_d.add_column('Annual Real Return (%) (Derived)', justify='right', style='dim')
        table_d.add_column('Annual Nominal Return (%) (Derived)', justify='right', style='dim')
        table_d.add_column('Max Drawdown (%)', justify='right', style='bold red')

        for index, row in df_means.iterrows():
            table_d.add_row(
                f"{int(row['Index'])}",
                row['Regime Label'],
                f"{row['FEATURE: ANNUAL Volatility (%) (Real)']:.2f}",
                f"{row['Annual Real Return (%) (Derived)']:.2f}",
                f"{row['Annual Nominal Return (%) (Derived)']:.2f}",
                f"{row['Max Drawdown (%)']:.2f}"
            )
        console.print(table_d)

      # --- 4.5 EXPORT CLASSIFICATION TO CSV (Nominal Returns + Regime ID) ---
        export_data = []

        for idx, row in df_quarterly.iterrows():
            # Determine the slice of daily data for this quarter
            start_loc = df.index.get_loc(idx)
            end_loc = start_loc + LOOKBACK_PERIOD
            if end_loc <= len(df):
                daily_slice = df.iloc[start_loc:end_loc]
            else:
                daily_slice = df.iloc[start_loc:]
    
            # Calculate mean nominal return for the quarter
            mean_nominal_return = daily_slice['Daily_Nominal_Returns'].mean() * TRADING_DAYS_YEAR
    
            export_data.append({
                'Quarter Start Date': idx.strftime('%Y-%m-%d'),
                'Mean Nominal Return (%)': mean_nominal_return,
                'Inferred Regime ID (0=Low, 2=High)': row['Inferred_Regime_VolRanked']
            })

        df_export = pd.DataFrame(export_data)
        output_file_name = 'regime_classification_nominal_returns.csv'
        df_export.to_csv(output_file_name, index=False)

        console.print(f"SUCCESS: Quarterly Nominal Return + Regime ID exported to {output_file_name}")


        # --- 5. VISUALIZATION ---

        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(16, 9))

        # Define colors for the three regimes
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # 0=blue, 1=orange, 2=green

        # --- CHANGE 2: Generalize label ---
        # Plot the daily REAL returns data for context
        df['Daily_Real_Returns'].plot(ax=ax, color='white', alpha=0.6, label='Investment Real Returns (Daily Log %)')
        ax.set_ylabel('Daily Real Log Returns (%)', color='white')

        # Shade the background according to the inferred regime using the daily-mapped data
        # 'Visual_Regime' now holds the volatility-ranked index (0, 1, 2) which is directly used for colors.
        for i in range(len(df) - 1):
            regime = df['Visual_Regime'].iloc[i]
            if pd.notna(regime):
                # Since Visual_Regime is the new 0, 1, 2 index, we use it directly as the index for colors.
                regime_index = int(regime) 
                if regime_index < NUM_REGIMES:
                    ax.axvspan(df.index[i], df.index[i+1], facecolor=colors[regime_index], alpha=0.2, linewidth=0)

        # Add dummy legend for regimes - sorted by volatility rank
        from matplotlib.patches import Patch
        # Create legend elements in the sorted order
        legend_elements = [Patch(facecolor=colors[data['final_index']], edgecolor='none', alpha=0.5, 
                                label=f"{data['final_index']}: {data['Label']}")
                        for data in sorted_regime_presentation_data]

        
        ax.legend(handles=legend_elements, loc='upper right', title='Inferred Volatility Regimes (0=Low, 2=High)')
        ax.set_title(f'Investment Regimes Inferred by Volatility (Real Returns)')
        ax.set_xlabel('Date')
        plt.tight_layout()
        plt.show()
