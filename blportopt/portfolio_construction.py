import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from blportopt.config import (
    ASSET_TICKERS,
    RF_COL,
)
from blportopt.covariance_estimator import portfolio_data
from blportopt.optimizer import PortoflioOptimizer



def empirical_rf_calculate(asset_type, freq=12):
    """
    Determine Excess Asset Returns (Historical), Standard Deviation, Covariance Matrix and Risk-Free Rates for all equities

    Parameters
    ----------

    asset_type : str
        Type of asset (fund/stock)
        
    freq : int
        Frequency of returns
    
    Returns
    -------

    freq_rf : float
        Average risk-free rate from historical 'RF' data

    """

    # --------- Compute Annual Returns, Covariance Matrix based on frequency of historical returns ------------ #
    asset_rf_data = portfolio_data(tickers=ASSET_TICKERS[asset_type])
    rf = asset_rf_data[RF_COL]

    print("-" * 50 + "Computing Average Risk-Free Returns of Assets" + "-" * 50)
    freq_rf = rf.mean() * freq

    return freq_rf


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



def efficient_frontier(mu, cov, rf, risk_aversion, method="volatility"):
    """
    Constructing Efficient Frontier by optimizing portfolios across a range of target returns

    Parameters
    ----------

    mu : pd.Series
        Average excess annual returns of assets within portfolio

    cov : pd.DataFrame
        Covariance Matrix

    rf : float
        Average risk-free rate from historical 'RF' data
    
    risk_aversion : float
        Investor risk appetite
    
    method: str
        Objective function to optimize
    
    Returns
    -------

    efport : pd.DataFrame
        Pandas dataframe with target returns, computed target volatilites from optimization process, and target sharpe ratios

    """
    target_returns = np.linspace(mu.min(), mu.max(), 100)
    tvols = []
    for tr in target_returns:
        
        port_optim = PortoflioOptimizer(mu=mu, cov=cov, tr=tr, rf=rf, risk_aversion=risk_aversion)
        
        opt_ef = port_optim.optimize(method=method)

        # Compute Volatiity
        tvols.append(port_optim.volatility(w=opt_ef["x"]))


    target_volatilities = np.array(tvols)

    efport = pd.DataFrame(
        {
            "targetrets": np.round_(100*target_returns, decimals=2),
            "targetvols": np.round_(100*target_volatilities, decimals=2),
            "targetsharpe": np.round_(target_returns/target_volatilities, decimals=2)
        }
    )

    return efport

def plot_efficient_frontier(efport_dict):

    plt.figure(figsize=(8,8))
    for item, efport in efport_dict.items():
        plt.scatter(efport["targetvols"], efport["targetrets"], label=item)
        maxSR_index = efport["targetsharpe"].argmax()
        plt.scatter(efport.loc[maxSR_index, ["targetvols"]], efport.loc[maxSR_index, ["targetrets"]], color='g', s=300, marker="*")
    plt.xlabel(r"Expected Volatilities ($\sigma$)")
    plt.ylabel(r"Expected Returns ($r$)")
    plt.grid(True)
    plt.legend(loc='best')
    plt.show()
