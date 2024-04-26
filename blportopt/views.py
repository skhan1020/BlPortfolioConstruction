import numpy as np
import pandas as pd
from blportopt.config import EARNINGS_FIELDS

from blportopt.data_utils import (
    EarningsReportLoader, 
)
from blportopt.portfolio_construction_BL import annual_excess_asset_returns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def get_dictionary_of_views(returns_dict):
    """
    Generate Absolute and Relative Views associated with every possible equity in the portfolio

    Parameters
    ----------

    returns_dict : dict
        Dictionary of expected returns predicted by ML model from earnings history
    
    Returns
    -------

    absolute_view_dict: dict
        dictionary of predicted returns for every equity
    
    relative_view_dict: dict
        dicionary of relative returns associated with every pair of equity

    """

    absolute_view_dict, relative_view_dict = dict(), dict()
    for i in returns_dict:
        for j in returns_dict:
            if i != j:
                # Absolute Views
                absolute_view_dict.update({str(i) + " returns" : returns_dict[i]})
                
                # Relative Views
                relative_view_dict.update({str(i) + " outperforms " + str(j) : returns_dict[i] - returns_dict[j]})
    

    return absolute_view_dict, relative_view_dict

def compute_matrices(investor_views, returns_dict, historical_returns):
    """
    Computes position matrix and return vector based on investor views and expected asset predictions 

    Parameters
    ----------

    investor_views : List[str]
        List of investor views used to upudate portfolio allocations
    
    returns_dict : dict
        Expected predicted returns from earnings history
    
    historical_returns : pd.Series
        Average historical returns of assets
    
    Returns
    -------
        P : np.ndarray
            position_matrix
        
        Q : np.array
            return_vector

    """
    # Obtain indices of every asset
    ticker_idx_dict = {k: idx for idx, k in enumerate(historical_returns.to_dict())}
    
    # Compute absolute & relative views
    absolute_view_dict, relative_view_dict = get_dictionary_of_views(returns_dict=returns_dict)

    # Initialize position matrix and return vector
    P = np.zeros((len(investor_views), len(returns_dict)))
    Q = np.zeros((len(investor_views)))

    for idx, investor_view in enumerate(investor_views):
        if investor_view in absolute_view_dict:
            
            ticker = investor_view.split(" returns")[0]
            ticker_idx = ticker_idx_dict[ticker]

            # Update P & Q matrices based on absolute views            
            P[idx][ticker_idx] = np.sign(absolute_view_dict[investor_view])
            Q[idx] = absolute_view_dict[investor_view]

        elif investor_view in relative_view_dict:
            
            ticker1 = investor_view.split(" outperforms ")[0]
            ticker2 = investor_view.split(" outperforms ")[-1]
            ticker1_idx, ticker2_idx = ticker_idx_dict[ticker1], ticker_idx_dict[ticker2]
            
            # Update P & Q matrices based on relative views
            P[idx][ticker1_idx], P[idx][ticker2_idx] = np.sign(relative_view_dict[investor_view]), -np.sign(relative_view_dict[investor_view])
            Q[idx] = relative_view_dict[investor_view]

    return P, Q
    


def generate_returns(data, ticker):
    """
    Predict Future Average Returns from Earnings Reports using Regression Model

    Parameters
    ----------

    ticker : str
        Ticker symbol for the equity used in forecasting future returns based on prior earnings

    Returns
    -------

    avg_predictions : float
        expected future returns 

    """
    # Earnings Report - Input Features; Future Returns - Target
    X, y = data[EARNINGS_FIELDS], data.loc[:, [ticker]].shift(-1)
    X_train, X_test, y_train, _ = train_test_split(X, y, shuffle=False, test_size=0.2)
    
    # Linear Regression model
    model = LinearRegression()

    # Model Fitting
    model.fit(X_train, y_train)
    
    # Model Predictions (Future Returns based on Earnings Reports)
    predictions = model.predict(X_test)

    # Average Future Returns
    avg_predictions = predictions.mean()

    return avg_predictions


def generate_positions(investor_views, historical_returns, tickers, from_file=True):
    """
    Function to Generate the Position and Return Matrices for the Likelihood Function in BL model

    Parameters
    ----------

    investor_views : List[str]
        List of Investor Views 
        Two possible investor types : 
            a) Absolute View : Company 1 returns
            b) Relative View : Company 1 outperforms Company 2
    
    historical_returns : pd.Series
        Average of historical returns of each quity within portfolio


    tickers : List[str]
        List of ticker symbols of equities included in portfolio

    from_file : bool (default True)
        Load earnings report from saved pickle file if True else collect earnings report from Alpha Vantage API and store in pickle file
    
    Returns
    -------

    position_matrix : np.ndarray
        Positions taken by investor based on Absolute & Relative Views

    return_vector : np.array
        Expected returns (absolute or relative) associated with each view of the investor

    """
    # Earnings Reports of All Assets in Portfolio
    equity_earnings_obj = EarningsReportLoader(tickers=tickers, from_file=from_file)
    
    # Quarterly Earnings Reports
    earnings_reports = equity_earnings_obj.get_earniings_history()

    # Excess Asset Returns (Historical)
    annual_stock_returns_data = annual_excess_asset_returns(tickers=tickers)


    ticker_predictions = dict()
    for ticker in tickers:
        
        # Earnings Report of chosen Equity
        ticker_earnings_reports = earnings_reports[earnings_reports["ticker"]==ticker]
        ticker_earnings_reports.drop(columns=["ticker"], inplace=True)
        
        # Historical Risk Premia of Equity
        ticker_returns = annual_stock_returns_data.loc[:, [ticker]]
        
        # Combine Equity Earnings Reports (Features) with Excess Returns (Target)
        ticker_df = pd.merge(ticker_earnings_reports, ticker_returns, how="inner", left_index=True, right_index=True)
        ticker_df.sort_index(inplace=True)

        # Future Returns based on Previous Month's Earnings Reports
        avg_predictions = generate_returns(data=ticker_df, ticker=ticker)
        ticker_predictions.update({ticker: avg_predictions})        

    # Generates Views    
    position_matrix, return_vector = compute_matrices(investor_views=investor_views, returns_dict=ticker_predictions, historical_returns=historical_returns)

    return position_matrix, return_vector    
        

    
    