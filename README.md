# robo advisor
robo advisor documentation

-AWGP_RoboAdvisor.xlsx: user interface. It has a short description of our robo advisor and users can input their information that helps construct their customized portfolio.

- roboadvisor.py:
1. grab users' input data from the excel workbook "AWGP_RoboAdvisor"
2. link to our ETFs, factors, FX database. Select ETFs with low correlations historically. Finally kept 15 non-ESG ETFs and 3 ESG ETFs.
3. set test period and calibration period for each rebalancing point.
4. calibrate the OLS regression model to get the VCV matrix and run Risk-Parity to obtain the optimal weights for each ETFs within each asset class, for each country.
5. output the portfolio value evolution, portfolio weighting, adjusted R-squared, timne series date and asset turnover.

- RP.py:
1. wrote the function RP(Q) that takes the VCV matrix as the input and output the optimal weightings.

- OLS.py:
1. constructed the OLS regression model in the function OLS(returns, factRet).
