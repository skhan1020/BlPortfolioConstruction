import numpy as np
import pandas as pd
from config import (
    RF_COL,
    FF_FILENAMES,
)
from data_utils import (
    FamaFrenchFactorDataLoader,
    EquityDataLoader,
)
from optimizer import (
    MeanVarOptimizer,
    MaxSharpeRatioOptimizer,
)

# Initial Portfolio Equities (Equities Determined from Optimal Portfolio Determined using French Fama Factor Model)
# optimal_portfolio_allocation = {'IBM': 0.2310020985134861, 'AMD': 0.2074646313673274, 'BAC': 0.1971430079162166, 'WHR': 0.18338721241903663, 'JPM': 0.18100304978393322}
PORTFOLIO_EQUITIES = ['IBM', 'AMD', 'BAC', 'WHR', 'JPM']

def excess_asset_returns():
    """
    Determine Excess Asset Returns (Historical) and Risk-Free Rates from Factor Data for all equities
    """
    print("-" * 50 + "Loading Time Series of Factors" + "-" * 50)
    famafrenchfactor = FamaFrenchFactorDataLoader()
    
    # Extract Fama-French (6) Factors data downloaded from library
    ff_data = famafrenchfactor.get_factor_data(filenames=FF_FILENAMES)
    ff_data = ff_data / 100

    # Extract Close Prices of each Stock
    stock_data_obj = EquityDataLoader(tickers=PORTFOLIO_EQUITIES)
    stock_close_data = stock_data_obj.get_history(price_type="Close")

    # Monthly Returns on Individual Stocks
    stock_returns = stock_data_obj.get_returns(stock_close_data)

    merged_stock_rf_data = pd.merge(stock_returns, ff_data[RF_COL], how="inner", left_index=True, right_index=True)

    # Determine Excess Stock Returns ( Subtract Risk Free Returns to Calculate Risk Premia Associated with Each Stock )
    for stock in PORTFOLIO_EQUITIES:
        merged_stock_rf_data[stock] = merged_stock_rf_data[stock] - merged_stock_rf_data[RF_COL]

    return merged_stock_rf_data


def annual_excess_asset_returns():
    """
    Function to compute annual returns (excess)
    """
    stock_rf_data = excess_asset_returns()

    stock_rf_data.drop(columns=["RF"], inplace=True)
    stock_rf_data.reset_index(inplace=True)
    stock_rf_data["Year"] = stock_rf_data["Date"].dt.year.astype(str)

    stock_rf_data1 = stock_rf_data.groupby(["Year"])[PORTFOLIO_EQUITIES].mean().reset_index()
    stock_rf_data1.set_index("Year", inplace=True)
    
    return stock_rf_data1


def portfolio_data(stock_rf_data, freq):
    """
    Function to generate annual returns (excess), standard dev, covariance, risk-free rate
    """
    stock_returns = stock_rf_data[PORTFOLIO_EQUITIES]
    rf = stock_rf_data[RF_COL]


    freq_returns = stock_returns.mean() * freq
    freq_stdev = stock_returns.std() * np.sqrt(freq)
    cov = stock_returns.cov() * freq
    freq_rf = rf.mean() * freq

    return freq_returns, freq_stdev, cov, freq_rf


def calc_optimal_portfolio_weights(mu, cov, rf, method, risk_aversion):
    """
    Calculate Optimal Portfolio Weights using Optimization Strategy
    """
    if method == "Mean-Variance":
        # Calculate Optimal Weights determined by constrained Mean-Variance Optimization 
        optimizer = MeanVarOptimizer(mu=mu, cov=cov, risk_aversion=risk_aversion)

    elif method == "Max Sharpe Ratio":
        optimizer = MaxSharpeRatioOptimizer(mu=mu, cov=cov, rf=rf)

    optimal_weights = optimizer.optimal_w

    return optimal_weights


    
