# Lifecycle-Retirement-Simulation

**Version 6.3** - Cythonized with Block Bootstrap Support

This script is designed to simulate possibility of outcomes to not only simulate how robust your portfolio is at sustaining withdrawals in retirement, but at what age you will be able to retire given an uncertain labor income. 



How it Works

Stage 1: The Destination

The script first calculates the portfolio size you need to achieve your desired spending goal at various retirement ages. It estimates this by running many "nested" simulations to find the principal required for a specified success rate.

Stage 2: The Journey

Next, the script runs thousands of "outer" simulations. In each one, it simulates your working life, including saving, market returns, and potential unemployment. It tracks your portfolio's growth until it reaches the principal goal, providing a distribution of possible retirement ages.

Key Features:

**Stochastic Modeling**: Uses a Bates jump-diffusion model for market returns (parametric) or block bootstrap from historical data, with normal distributions for inflation, salary growth and more.

**Cython Acceleration**: Compiled Cython extensions provide 10-50x speedup on computational hotspots (automatic fallback to Python if unavailable).

**Block Bootstrap Option**: Can use historical data blocks instead of parametric model to preserve historical correlations between returns and inflation.

**Monthly Time Steps**: Simulates on a monthly basis for more granular and accurate modeling.

**Unemployment Modeling**: Realistic unemployment dynamics with exit probability coupled to market returns.

**Detailed Outputs**: Generates summary tables and a series of plots that visualize your results.

**Data Export**: Exports detailed simulation data to CSV files for analysis and debugging.

## Setup

### Installation
Install the required libraries:
```bash
pip install numpy pandas tqdm rich matplotlib cython
```

**Note**: Cython is optional but highly recommended for performance. The script will automatically fall back to pure Python if Cython modules are not available.

### For Google Colab
1. Run: `!pip install cython numpy pandas tqdm rich matplotlib`
2. Then run the script (Cython will compile inline automatically)


The script's behavior is controlled by the User inputs / parameters section near the top. You can customize the simulation by changing these values:

initial_age: Your starting age.

death_age: The age the simulation runs until.

initial_portfolio: Your current savings.

annual_income_real: Your current annual income, in today's dollars.

spending_real: Your target annual spending in retirement, in today's dollars.

num_outer: The total number of full lifecycle simulations to run. A higher number provides a more statistically robust result but is more computationally intense.

num_nested: The number of simulations used in Stage 1 to calculate the required principal. A higher number provides a more accurate principal lookup table, but again more computationally intense.

social_security_real: The annual Social Security benefit you expect to receive, in today's dollars.

include_social_security: A toggle to include Social Security benefits in the calculation.

You can also adjust the asset assumptions and portfolio weights to match your personal investment strategy. In this example, the mean returns are forecasted, but the variances and covarainces are sourced from Testfol.io. Testfol.io is a good source.

Jump Intensity, Jump Mean, Jump Std Dev are derrived from historical data. Here I fit the historical data to a Bates model. This script will also be provided in this Repo, and again I source a backtest of the portfolio from Testfol.io. If you want, you can alter these parameters to be whatever you would like. 

The simulation is currently using a **Fixed Spending Rule**, where a constant, inflation-adjusted amount is withdrawn from the portfolio each month in retirement.

**Version 6.3 Improvements:**
- Cython acceleration for 10-50x performance improvement
- Block bootstrap option to use historical data instead of parametric model
- Monthly time steps (upgraded from annual in version 4)
- Enhanced unemployment modeling with market-return-coupled exit probabilities
- Improved multiprocessing support for parallel simulations
- Better error handling and fallback mechanisms 

Over time I plan on upgrading this to use an Amortization based withdrawal strategy that will allow some variability in spending. 

This change will require major overhauls on how we measure "success", and how we are running the simulation. Below are some of the improvements I plan on making to the script. No particular order.

1. ~~Allow the Monte Carlo simulation to switch between regimes. Each regime will have different parameters.~~ *(Partially implemented in parametric model modules - regime switching available but not yet integrated into main simulation)*
2. Upgrade from Fixed Withdrawal strategy to an Amortization Based Withdrawal (ABW) strategy recalculated annually (each step). Perhaps use binary search to estimate what principal is needed to sustain an average real spending around $X, at Y age with Z% success rate.
3.  Add forecast error on expected return assumptions for ABW rather than assuming perfect hindsight, or assuming static mean specified in beginning. Perhaps by making return assumptions at time t positively but imperfectly correlated with realized return at t+1. Thinking 0.4 by default.
4. Calculating Utility of consumption and bequest during withdrawal period. Adding real bequest to ABW strategy.
5. Proper estimation on CRRA, strength of bequest motive, and bequest curvature parameter.
6. Percentage of times the portfolio was able to sustain above $X real during the withdrawal period.
7. Simulate Mortality of a heterosexual couple so we properly account for longevity risk
8. Rather than savings rate, have an "expenses level" where those expenses are financed first, and savings are what is left over.
9. Incorporate Taxes and Fees (Testfol.io likely already incorporates fees, if you are sourcing from them)
10. Upgrade to GARCH Model for inflation. 
11. Option to purchase an annuity at retirement
12. Adding the ability to buy and hold bonds to maturity rather than relying on ETF's so we can see how duration matching impacts utility of consumption and bequest
13. Adding inflation indexed bond
14. Adding Glide path. Perhaps contingent on how far investor is from our estimated "FI/RE" number.
15. Upgrade to simulate constituent assets individually, rather than simulating the entire portfolio.
