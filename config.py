# Portfolio of Stocks
STOCK_TICKERS = ['AAPL', 'IBM','PG', 'GE', 'AMD', 'WMT', 'BAC', 'T', 'XOM', 'RRC', 'BBY', 'PFE', 'JPM', 'C','MCD', 'KO', 'JNJ', 'WHR', 'MMM', 'GIS']

# Market Index Ticker
MARKET_TICKER = "SPY"

# Equity Start/Stop Dates
EQ_START_DATE = "1963-07-01"
EQ_END_DATE = "2023-12-01"

# Equity Price Intervals
INTERVAL = '1mo' 

# Fama-French Data Start Date
FF_START_DATE = "1-1-1926"

# Fama-French Factor List of FileNames
FF_FILENAMES = ["F-F_Research_Data_5_Factors_2x3", "F-F_Momentum_Factor"]

# Fama-French Factors (6)
FF_FACTORS = ['Mkt-RF','SMB','HML','RMW','CMA','Mom']

# Risk-Free Rate Variable
RF_COL = "RF"

# Sliding Window for Rolling Beta Estimation
WINDOW = 60

# Flag
ROLLING=True

# Initial Capital
INITIAL_CAPITAL=1000

# Top N Stocks Selected in Alpha Trading Strategy
TOPN=5

# Strategies based on Individual Factors
STRATEGIES = [['Alpha'], ['Mkt-RF'], ['SMB'], ['HML'], ['RMW'], ['CMA'], ['Mom']]

# All Possible Factor Combinations
ALL_STRATEGIES = list()
for i in range(0, len(STRATEGIES)):
    strategy_list = [STRATEGIES[i][0]]
    ALL_STRATEGIES.append(strategy_list)
    for j in range(i+1, len(STRATEGIES)):
        strategy_list = strategy_list + [STRATEGIES[j][0]]
        ALL_STRATEGIES.append(strategy_list)

# Earnings Report From Alpha Vantage API
FUNCTION = 'EARNINGS'
BASE_URL = 'https://www.alphavantage.co/query?'

# List of fields used in Earnings Report of Each Equity
EARNINGS_FIELDS = [
     'reportedEPS','estimatedEPS','surprise','surprisePercentage'
]

# Alpha Vantage API Key
API_KEY = "O16ZZ2O1MIS9KZLJ"

# Equilibrium Sharpe Ratio for Black-Litterman Model
SHARPE_RATIO_MKT = 0.5

# Possible Risk Aversion Values
RISK_AVERSION_VALUES = [2.24, ]