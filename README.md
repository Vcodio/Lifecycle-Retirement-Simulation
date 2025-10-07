# Lifecycle-Retirement-Simulation
This script is designed to simulate possibility of outcomes to not only simulate how robust your portfolio is at sustaining withdrawals in retirement, but at what age you will be able to retire given an uncertain labor income. 



How it Works

Stage 1: The Destination

The script first calculates the portfolio size you need to achieve your desired spending goal at various retirement ages. It determines this by running many "nested" simulations to find the principal required for a specified success rate.

Stage 2: The Journey

Next, the script runs thousands of "outer" simulations. In each one, it simulates your working life, including saving, market returns, and potential unemployment. It tracks your portfolio's growth until it reaches the principal goal, providing a distribution of possible retirement ages.

Key Features:

Stochastic Modeling: Uses a Merton Jump Diffusion model for market returns and normal distributions for inflation, salary growth and more.

Detailed Outputs: Generates summary tables and a series of plots that visualize your results.

Data Export: Exports detailed simulation data to CSV files for analysis and debugging.

Setup
Install the required libraries:
pip install numpy pandas tqdm rich matplotlib


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

Jump Intensity, Jump Mean, Jump Std Dev are derrived from historical data. Here I fit the historical data to a Merton Jump Diffusion model. This script will also be provided in this Repo, and again I source a backtest of the portfolio from Testfol.io. If you want, you can alter these parameters to be whatever you would like. Here we have a large jump intensity with a low jump mean, but a low jump intensity and high jump mean might be better suited for you. 

The simulation is currently using a Fixed Spending Rule, where a constant, inflation adjusted amount is withdrawn from the portfolio each year in retirement. 

Over time I plan on upgrading this to use an Amortization based withdrawal strategy that will allow some variability in spending. 

This change will require major overhauls on how we measure "succsss", and how we are running the simulation. Below are some of the improvements I plan on making to the script. No particular order.

1. Upgrade from Fixed Withdrawal strategy to an Amortization Based Withdrawal (ABW) strategy recalculated annually (each step). Perhaps use binary search to estimate what principal is needed to sustain an average real spending around $X, at Y age with Z% success rate.
2.  Add forecast error on expected return assumptions for ABW rather than assuming perfect hindsight, or assuming static mean specified in beginning. Perhaps by making return assumptions at time t positively but imperfectly correlated with realized return at t+1. Thinking 0.4 by default.
3. Calculating Utility of consumption and bequest during withdrawal period. Adding real bequest to ABW strategy.
4. Proper estimation on CRRA, strength of bequest motive, and bequest curvature parameter.
5. Percentage of times the portfolio was able to sustain above $X real during the withdrawal period.
6. Simulate Mortality of a heterosexual couple so we properly account for longevity risk
7. Rather than savings rate, have an "expenses level" where those expenses are financed first, and savings are what is left over.
8. Incorporate Taxes and Fees (Testfol.io likely already incorporates fees, if you are sourcing from them)
9. Upgrade to GARCH Model for inflation. 
10. Option to purchase an annuity at retirement
11. Adding the ability to buy and hold bonds to maturity rather than relying on ETF's so we can see how duration matching impacts utility of consumption and bequest
12. Adding inflation indexed bond
13. Adding Glide path. Perhaps contingent on how far investor is from our estimated "FI/RE" number.
14. Upgrade to simulate constiuent assets individually, rather than simulating the entire portfolio.
