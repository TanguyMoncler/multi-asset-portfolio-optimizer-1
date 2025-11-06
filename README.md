//--------NOTE------------------------------------//

Ceci est un projet personnel.

Mon objectif est de montrer mes compétences en codage python
mais aussi en compréhension de l'optimisation de portefeuille.
Je l'ai réalisé seul en prenant mon temps pour faire 
un fichier README explicatif de l'intégralité du programme pour 
le rendre lisible et compréhensible par tous.

Tanguy Moncler

//--------SYNTHESE--------------------------------//
Le programme récupère les prix daily d'un univers de tickers.
(ETF . BONDS . COMMODITIES)

Il génère un portefeuille selon les contraintes entrées par l'utilisateur.
Il fait un backtest et rebalance le pf chaque semaine sur une plage 
temporelle définie par l'utilisateur.


//--------EXECUTION DU PROGRAMME------------------//

-> Rendez-vous sur Google Colab ou Python en local sur votre ordinateur.
-> Copier la ligne de code suivante et l'exécuter dans votre notebook:

! pip install pandas numpy yfinance matplotlib cvxpy

-Coller l'intégralité du code compris dans le document "multi-asset-portfolio-optimizer.py"
-Exécuter le programme et selectionner vos contraintes.



//--------PORTFOLIO OPTIMIZER-------------------//

>>> Univers US <<<
ETF: SPY ; VEA ; EEM
BONDS: AGG ; BNDX
COMMODITIES : GLD ; DBC

>>> Univers EU <<<
ETF: CSPX.L, VEVE.L, EIMI.L
BONDS: IEAC.L, IGLA.L
COMMODITIES: SGLN.L, CMOD.L

>>> Contraintes proposées à l'utilisateur<<<
-> Long only ou non ?
-> Cap poids ?
-> Caps par groupe ?
-> Cible de volatilité ?
-> Tracking-error maximum par rapport au benchmark ?
-> Turnover maximum ?
-> ESG score minimum ?


Si l'utilisateur choisi des ETF US alors le Bench est : 
60 % SPY (S&P 500)
40 % AGG (US Aggregate Bonds)

Si l'utilisateur choisi des ETF UCITS Europe alors le Bench est :
60 % CSPX.L (iShares Core S&P 500 UCITS ETF)
40 % IEAC.L (iShares Euro Aggregate Bond UCITS ETF)




//----------------------------------------------///

