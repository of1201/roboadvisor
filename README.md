# AWGP robo advisor
robo advisor documentation

- risk lab.ipynb: this jupyter notebook python file is the main program that contains all the code for our system implementation as well as report generation. 
  * It also contains two blocks of code that are for backtesting with some parameters preset instead of imported from our user interface "AWGP_RoboAdvisor.xlsx". When running the main system implementation, the two blocks of code for backtesting should be commmented out
  * It contains the code for risk models, sensitivity/scenario analysis, generating performance & risk metrics, as well as exporting the report metrics and graphs to the user interface "report.xlsx".
  * Portfolio manager  can set the asset allocation model and ETFs tickers based on our model selection and backtesting result. In backtesting, we can also set which model to test at the end of block.

- AWGP_RoboAdvisor.xlsx: user interface. It has a short description of our robo advisor and users can input their information that helps construct their customized portfolio.

- Report.xlsx: It is the Excel template that we use to let our system automatically export the client report into. The cell and page format is preset.
  * Every time you run our program "risk lab.ipynb", it write the output to "Report.xlsx". So before you run the program again, you should first prepare the empty "Report.xlsx" again so the program outputs the results to it smoothly.

- Benchmark.xlsx: Contains the historical price (2007-01-01 to 2021-05-31) of 4 benchmarks that represent 4 asset classes - equity, fixed income, real estate and commodity - respectively. 

- NonESG_Prices.xlsx: Contains the historical price (2007-01-01 to 2021-05-31) of 36 pre-selected non-ESG ETFs that are from the 4 asset classes mentioned above. These ETFs are in CAD or USD and some of them have international exposure.

- ESG_Prices.xlsx: Contains the historical price (2007-01-01 to 2021-05-31) of 4 pre-selected ESG ETFs. They are all equity ETFs.

- FF_5_Factor.csv / FF_5_Factor_all.csv: Contains the historical price for the Fama-French five factors. FF_5_Factor.csv has prices from 2007-01-01 to 2021-05-31 and FF_5_Factor_all.csv has prices from 1963-08-01 to 2021-05-31.

- Historical_Data.xlsx: Contains older price data for the 36 non-ESG ETFs and the 4 ESG ETFs (from 1998-01-01 to 2021-05-31). 

- Macro_Data.xlsx: Contains the historical price (2007-01-01 to 2021-05-31) of 16 macroeconomic factors such as unemployment rate, crude oil price, and housing data.

- USD_CAD_Historical_Data.csv: It contains the historical USD to CAD foreign exchange rates data (2007-01-01 to 2021-05-31).

- beta.xlsx: This file shows the beta of all the ETFs that we finally select to be included in our portfolio.

- sensitivity.xlsx: This file contains our sensitivity/stress backtesting results for all 3 of our asset allocation models: CVaR, RP and robust MVO.

