# Lifecycle-Retirement-Simulation
This script is designed to simulate possibility of outcomes to not only simulate how robust your portfolio is at sustaining withdrawals in retirement, but at what age you will be able to retire given an uncertain labor income. 

The simulation is based on a Fixed Spending Rule, where a constant, inflation adjusted amount is withdrawn from the portfolio each year in retirement. Over time I plan on upgrading this to use an Amortization based withdrawal strategy that will 

How it Works
Stage 1: The Destination
The script first calculates the portfolio size you need to achieve your desired spending goal at various retirement ages. It determines this by running many "nested" simulations to find the principal required for a specified success rate.

Stage 2: The Journey
Next, the script runs thousands of "outer" simulations. In each one, it simulates your working life, including saving, market returns, and potential unemployment. It tracks your portfolio's growth until it reaches the principal goal, providing a distribution of possible retirement ages.

Key Features
Stochastic Modeling: Uses a Merton Jump Diffusion model for market returns and normal distributions for inflation and salary growth.

Detailed Outputs: Generates summary tables and a series of plots that visualize your results.

Data Export: Exports detailed simulation data to CSV files for deeper analysis.

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
