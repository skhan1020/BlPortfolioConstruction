import pandas as pd
import numpy as np
from blportopt.config import (
    ASSET_TICKERS,
    EQ_START_DATE,
    EQ_END_DATE,
    RF_COL,
    FF_FACTORS,
    MARKET_TICKER,
    FF_FILENAMES,
)
from blportopt.data_utils import (
    EquityDataLoader,
    FamaFrenchFactorDataLoader,
    get_data,
)
from blportopt.fama_french_model import FamaFrenchModel


def portfolio_data(tickers, start=EQ_START_DATE, end=EQ_END_DATE):
    """
    Determine Excess Asset Returns and Risk-Free Rates for all equities

    Parameters
    ----------

    tickers : List[str]
        List of assets (ticker symbols)
    
    freq : int
        Frequency of returns
    
    Returns
    -------
    
    asset_rf_data : pd.DataFrame
        Average excess annual returns of assets + risk-free returns within portfolio
    """
    print("-" * 50 + "Loading Time Series of Factors" + "-" * 50)
    famafrenchfactor = FamaFrenchFactorDataLoader()
    
    # Extract Fama-French (6) Factors data downloaded from library
    ff_data = famafrenchfactor.get_factor_data(filenames=FF_FILENAMES)
    ff_data = ff_data / 100

    # Extract Close Prices of each Asset (Fund/Stock)
    asset_data_obj = EquityDataLoader(tickers=tickers, start=start, end=end)
    asset_close_data = asset_data_obj.get_history(price_type="Close")

    # Monthly Returns on Individual Assets (Funds/Stocks)
    asset_returns = asset_data_obj.get_returns(data=asset_close_data)

    asset_rf_data = pd.merge(asset_returns, ff_data[RF_COL], how="inner", left_index=True, right_index=True)

    # Determine Excess Stock Returns ( Subtract Risk Free Returns to Calculate Risk Premia Associated with Each Stock )
    for asset in tickers:
        asset_rf_data[asset] = asset_rf_data[asset] - asset_rf_data[RF_COL]
    
    asset_rf_data.reset_index(inplace=True)

    return asset_rf_data


def empirical_cov_calculate(asset_type, freq=12):
    """
    Determine Excess Asset Returns (Historical), Empirical Covariance Matrix

    Parameters
    ----------

    asset_type : str
        Type of asset (fund/stock)
        
    freq : int
        Frequency of returns
    
    Returns
    -------
    
    mu_emp : pd.Series
        Average excess annual returns of assets within portfolio
        
    cov_emp : pd.DataFrame
        Empirical Covariance Matrix obtained from Historical Returns
    """

    empirical_asset_returns = portfolio_data(tickers=ASSET_TICKERS[asset_type])[ASSET_TICKERS[asset_type]]
    
    mu_emp = empirical_asset_returns.mean() * freq
    cov_emp = empirical_asset_returns.cov() * freq

    return mu_emp, cov_emp


def factor_cov_calculate(asset_type, freq=12):
    """
    Determine Excess Asset Returns (Factor Model), Factor Model Evaluated Covariance Matrix

    Parameters
    ----------

    asset_type : str
        Type of asset (fund/stock)
        
    freq : int
        Frequency of returns
    
    Returns
    -------
    
    mu_f : pd.Series
        Average excess annual returns of assets within portfolio obtained from factor analysis
        
    cov_f : pd.DataFrame
        Factor Model generated Covariance Matrix
    """

    print("-" * 50 + "Computing Covariance Matrix based on Factor Loadings" + "-" * 50)


    # Train Fama-French Factor Model -- Factor Data (X), Asset Returns (y_train)
    ff_data, _, _, asset_returns, _ = get_data(asset_tickers=ASSET_TICKERS[asset_type], market_ticker=MARKET_TICKER, asset_type=asset_type)
    
    factor_data = ff_data[FF_FACTORS]
    alpha_coeff = pd.DataFrame(data=[1]*len(ff_data), columns=['const'], index=ff_data.index.tolist())
    factor_data = pd.concat([alpha_coeff, factor_data], axis=1)
    X = factor_data.to_numpy()


    computed_asset_returns = pd.DataFrame()
    

    for asset in ASSET_TICKERS[asset_type]:
        
        # Instantiate the Fama-French Factor model
        model = FamaFrenchModel(asset=asset)
        
        # Fitting of model
        model.fit(asset_data=asset_returns, ff_data=ff_data)

        # Compute Factor Loadings
        factor_loadings = model.params.mean().to_numpy()

        print("-" * 30 + f"Asset returns Computed for Asset {asset}" + "-" * 30)

        factor_returns = pd.DataFrame(data=X @ factor_loadings)
        factor_returns.columns = [asset]

        computed_asset_returns = pd.concat([computed_asset_returns, factor_returns], axis=1)

    mu_f = asset_returns.mean() * freq
    cov_f = asset_returns.cov() * freq

    return mu_f, cov_f

        