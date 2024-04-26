import numpy as np
import pandas as pd
from blportopt.config import (
    RF_COL,
    FF_FILENAMES,
)
from blportopt.data_utils import (
    FamaFrenchFactorDataLoader,
    EquityDataLoader,
)
from blportopt.optimizer import (
    MeanVarOptimizer,
    MaxSharpeRatioOptimizer,
)

# Initial Portfolio Equities (Equities Determined from Optimal Portfolio Determined using French Fama Factor Model)
PORTFOLIO_EQUITIES = ['IBM', 'AMD', 'BAC', 'WHR', 'JPM']

def excess_asset_returns(tickers):
    """
    Determine Excess Asset Returns (Historical) and Risk-Free Rates from Factor Data for all equities

    Parameters
    ----------

    tickers : List[str]
        List of assets (ticker symbols)
    
    Returns
    -------

    merged_stock_rf_data : pd.DataFrame
        excess asset returns for each equity included in the portfolio
    """
    print("-" * 50 + "Loading Time Series of Factors" + "-" * 50)
    famafrenchfactor = FamaFrenchFactorDataLoader()
    
    # Extract Fama-French (6) Factors data downloaded from library
    ff_data = famafrenchfactor.get_factor_data(filenames=FF_FILENAMES)
    ff_data = ff_data / 100

    # Extract Close Prices of each Stock
    stock_data_obj = EquityDataLoader(tickers=tickers)
    stock_close_data = stock_data_obj.get_history(price_type="Close")

    # Monthly Returns on Individual Stocks
    stock_returns = stock_data_obj.get_returns(stock_close_data)

    merged_stock_rf_data = pd.merge(stock_returns, ff_data[RF_COL], how="inner", left_index=True, right_index=True)

    # Determine Excess Stock Returns ( Subtract Risk Free Returns to Calculate Risk Premia Associated with Each Stock )
    for stock in tickers:
        merged_stock_rf_data[stock] = merged_stock_rf_data[stock] - merged_stock_rf_data[RF_COL]

    return merged_stock_rf_data


def annual_excess_asset_returns(tickers):
    """
    Function to compute annual returns (excess)

    Parameters
    ----------

    tickers : List[str]
        List of assets (ticker symbols)

    Returns
    -------

    stock_rf_data1 : pd.DataFrame
        Dataframe of excess annual returns of assets
    """
    stock_rf_data = excess_asset_returns(tickers=tickers)

    stock_rf_data.drop(columns=["RF"], inplace=True)
    stock_rf_data.reset_index(inplace=True)
    stock_rf_data["Year"] = stock_rf_data["Date"].dt.year.astype(str)

    stock_rf_data1 = stock_rf_data.groupby(["Year"])[tickers].mean().reset_index()
    stock_rf_data1.set_index("Year", inplace=True)
    
    return stock_rf_data1


def portfolio_data(stock_rf_data, tickers, freq):
    """
    Function to generate annual returns (excess), standard dev, covariance, risk-free rate

    Parameters
    ----------
    
    stock_rf_data : pd.DataFrame
        Historical Excess Annual Returns
    
    tickers : List[str]
        List of assets (ticker symbols)
    
    freq : int
        Frequency of returns
    
    Returns
    -------
    
    freq_returns : pd.Series
        Average excess annual returns of assets within portfolio
    
    freq_stdev : pd.Series
        Std. Dev of excess annual returns of assets within portfolio
    
    cov : pd.DataFrame
        Covariance Matrix
    
    freq_rf : float
        Average risk-free rate from historical 'RF' data
    """
    stock_returns = stock_rf_data[tickers]
    rf = stock_rf_data[RF_COL]


    freq_returns = stock_returns.mean() * freq
    freq_stdev = stock_returns.std() * np.sqrt(freq)
    cov = stock_returns.cov() * freq
    freq_rf = rf.mean() * freq

    return freq_returns, freq_stdev, cov, freq_rf


def calc_optimal_portfolio_weights(mu, cov, rf, method, risk_aversion):
    """
    Calculate Optimal Portfolio Weights using Optimization Strategy

    Parameters
    ----------

    mu : pd.Series
        Average excess annual returns of assets within portfolio

    cov : pd.DataFrame
        Covariance Matrix

    rf : float
        Average risk-free rate from historical 'RF' data
    
    method : str
        Optimization method
    
    risk_aversion : float
        Investor risk appetite

    Returns
    -------

    optimal_weights : np.array
        Array of optimal portoflio allocations
    """
    if method == "Mean-Variance":
        # Calculate Optimal Weights determined by constrained Mean-Variance Optimization 
        optimizer = MeanVarOptimizer(mu=mu, cov=cov, risk_aversion=risk_aversion)

    elif method == "Max Sharpe Ratio":
        optimizer = MaxSharpeRatioOptimizer(mu=mu, cov=cov, rf=rf)

    optimal_weights = optimizer.optimal_w

    return optimal_weights


    

