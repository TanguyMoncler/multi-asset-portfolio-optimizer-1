//--------NOTE------------------------------------//

This is a personal project.

My goal is to demonstrate my skills in Python coding
as well as my understanding of portfolio optimization.
I completed it on my own, taking the time to create
a detailed README file explaining the entire program,
so it can be readable and understandable by anyone.

Tanguy Moncler


//--------SUMMARY--------------------------------//

The program retrieves daily prices from a universe of tickers
(ETF . BONDS . COMMODITIES).

It generates a portfolio according to the constraints entered by the user.
It performs a backtest and rebalances the portfolio each week
over a time period defined by the user.


//--------PROGRAM EXECUTION----------------------//

-> Go to Google Colab or run Python locally on your computer.  
-> Copy and execute the following line of code in your notebook:

! pip install pandas numpy yfinance matplotlib cvxpy

-> Paste the entire code contained in the file “multi-asset-portfolio-optimizer.py”  
-> Run the program and select your constraints.


//--------PORTFOLIO OPTIMIZER-------------------//

US Universe  
>>> ETF: SPY ; VEA ; EEM  
>>> BONDS: AGG ; BNDX  
>>> COMMODITIES: GLD ; DBC  

EU Universe  
>>> ETF: CSPX.L ; VEVE.L ; EIMI.L  
>>> BONDS: IEAC.L ; IGLA.L  
>>> COMMODITIES: SGLN.L ; CMOD.L  

The number of securities is kept small, otherwise the code would take too long to run.

>>> Constraints proposed to the user <<<  
-> Long-only or not?  
-> Maximum weight per asset?  
-> Group caps?  
-> Target volatility?  
-> Maximum tracking error vs benchmark?  
-> Maximum turnover?  
-> Minimum ESG score?  

If the user chooses US ETFs, then the benchmark is:  
60% SPY (S&P 500)  
40% AGG (US Aggregate Bonds)

If the user chooses UCITS Europe ETFs, then the benchmark is:  
60% CSPX.L (iShares Core S&P 500 UCITS ETF)  
40% IEAC.L (iShares Euro Aggregate Bond UCITS ETF)


//----------------------------------------------//
