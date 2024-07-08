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
    PortoflioOptimizer,
    MeanVarOptimizer,
    MaxSharpeRatioOptimizer,
)

def portfolio_data(tickers, freq=12):
    """
    Determine Excess Asset Returns (Historical) and Risk-Free Rates from Factor Data for all equities

    Parameters
    ----------

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
    print("-" * 50 + "Loading Time Series of Factors" + "-" * 50)
    famafrenchfactor = FamaFrenchFactorDataLoader()
    
    # Extract Fama-French (6) Factors data downloaded from library
    ff_data = famafrenchfactor.get_factor_data(filenames=FF_FILENAMES)
    ff_data = ff_data / 100

    # Extract Close Prices of each Asset (Fund/Stock)
    asset_data_obj = EquityDataLoader(tickers=tickers)
    asset_close_data = asset_data_obj.get_history(price_type="Close")

    # Monthly Returns on Individual Assets (Funds/Stocks)
    asset_returns = asset_data_obj.get_returns(data=asset_close_data)

    asset_rf_data = pd.merge(asset_returns, ff_data[RF_COL], how="inner", left_index=True, right_index=True)

    # Determine Excess Stock Returns ( Subtract Risk Free Returns to Calculate Risk Premia Associated with Each Stock )
    for asset in tickers:
        asset_rf_data[asset] = asset_rf_data[asset] - asset_rf_data[RF_COL]

    # asset_rf_data.drop(columns=[RF_COL], inplace=True)
    asset_rf_data.reset_index(inplace=True)
    asset_rf_data["Year"] = asset_rf_data["Date"].dt.year.astype(str)

    # --------- Compute Annual Returns, Covariance Matrix based on frequency of historical returns ------------ #
    asset_returns = asset_rf_data[tickers]
    rf = asset_rf_data[RF_COL]


    freq_returns = asset_returns.mean() * freq
    freq_stdev = asset_returns.std() * np.sqrt(freq)
    cov = asset_returns.cov() * freq
    freq_rf = rf.mean() * freq


    return freq_returns, freq_stdev, cov, freq_rf
     # annual_asset_returns = asset_rf_data.groupby(["Year"])[tickers].mean().reset_index()
    # monthly_asset_returns.set_index("Year", inplace=True)

    # return monthly_asset_returns



def calc_optimal_portfolio_weights(mu, cov, rf, tr, risk_aversion, method):
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
    
    tr: float
        Target return of Portfolio
    
    method : str
        Optimization method
    
    risk_aversion : float
        Investor risk appetite

    Returns
    -------

    optimal_weights : np.array
        Array of optimal portoflio allocations
    """

    port_optim = PortoflioOptimizer(mu=mu, cov=cov, tr=tr, rf=rf, risk_aversion=risk_aversion)
        
    optimal_weights = port_optim.optimize(method=method)['x']

    return optimal_weights



def efficient_frontier(mu, cov, rf, risk_aversion, method):

    target_returns = np.linspace(mu.min(), mu.sum(), 100)
    tvols = []
    for tr in target_returns:
        
        port_optim = PortoflioOptimizer(mu=mu, cov=cov, tr=tr, rf=rf, risk_aversion=risk_aversion)
        
        opt_ef = port_optim.optimize(method=method)
        tvols.append(opt_ef["fun"])


    target_volatilities = np.array(tvols)

    efport = pd.DataFrame(
        {
            "targetrets": np.round_(100*target_returns, decimals=2),
            "targetvols": np.round_(100*target_volatilities, decimals=2),
            "targetsharpe": np.round_(target_returns/target_volatilities, decimals=2)
        }
    )

    return efport