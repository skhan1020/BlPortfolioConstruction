import numpy as np
import pandas as pd
from blportopt.config import EARNINGS_FIELDS

from blportopt.data_utils import (
    EarningsReportLoader, 
)
from blportopt.portfolio_construction_BL import (
    PORTFOLIO_EQUITIES,
    annual_excess_asset_returns,
)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


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


def generate_positions(tickers=PORTFOLIO_EQUITIES, from_file=True):
    """
    Function to Generate the Position and Return Matrices for the Likelihood Function in BL model

    Parameters
    ----------

    tickers : List[str]
        List of ticker symbols of equities included in portfolio

    
    Returns
    -------

    position_matrix : np.ndarray
        Positions taken by investor based on Absolute & Relative Views

    return_vector : np.array
        Expected confidence scores associated with each view of the investor
    """
    # Earnings Reports of All Assets in Portfolio
    equity_earnings_obj = EarningsReportLoader(tickers=PORTFOLIO_EQUITIES, from_file=from_file)
    
    # Quarterly Earnings Reports
    earnings_reports = equity_earnings_obj.get_earniings_history()

    # Excess Asset Returns (Historical)
    annual_stock_returns_data = annual_excess_asset_returns()


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

    # Relative View 1:     
    # By How many basis points is JP Morgan expected to outperform Bank of America Corp.
    view1 = {"JPM outperforms BAC": ticker_predictions["JPM"] - ticker_predictions["BAC"]}
    

    # Absolute View 2:
    # What are the "average" absolute excess returns of IBM and JP Morgan
    view2 = {"JPM expected to generate returns": ticker_predictions["JPM"]}

    # Relative View 3:
    # By how much will AMD outperform WHR
    view3 = {"AMD outperforms WHR": ticker_predictions["AMD"] - ticker_predictions["WHR"]}

    # Generate Position (P) and Return (Q) Matrices
    position_matrix = np.array(
        (
        [0, 0, np.sign(-view1["JPM outperforms BAC"]), 0, np.sign(view1["JPM outperforms BAC"])], 
        [0, 0, 0, 0, np.sign(view2["JPM expected to generate returns"])], 
        [0, np.sign(view3["AMD outperforms WHR"]), 0, np.sign(-view3["AMD outperforms WHR"]), 0]
        )
    )

    return_vector = np.array([view1["JPM outperforms BAC"], view2["JPM expected to generate returns"], view3["AMD outperforms WHR"]])

    return position_matrix, return_vector    
        

    
    