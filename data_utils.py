import os
import pandas_datareader as pdr
import pandas as pd
import yfinance as yf
from config import (
    EQ_START_DATE,
    EQ_END_DATE,
    INTERVAL,
    FF_START_DATE,
    FF_FILENAMES,
    STOCK_TICKERS,
    MARKET_TICKER,
    BASE_URL,
    FUNCTION,
    API_KEY,
    EARNINGS_FIELDS,
)
import requests
import pickle

class EquityDataLoader:
    """
    Class to Load Historical Prices of Equities (tickers) from Yahoo Finance API. Calculates Returns 
    """

    def __init__(self, tickers, start=EQ_START_DATE, end=EQ_END_DATE, interval=INTERVAL):
        self.tickers = tickers
        self.start = start
        self.end = end
        self.interval = interval
    
    def get_history(self, price_type):
        """
        Get Historical Prices (Open/Close)
        """

        data = pd.DataFrame()
        for ticker in self.tickers:
            yf_ticker = yf.Ticker(ticker)
            yf_ticker_history = yf_ticker.history(start=self.start, end=self.end, interval=self.interval)
            ticker_data = pd.DataFrame(yf_ticker_history[price_type])
            ticker_data.reset_index(inplace=True)
            ticker_data.columns = ['Date', ticker]
            if not len(data):
                data = ticker_data
            else:
                data = pd.merge(data, ticker_data, how="inner", on="Date")

        data.set_index('Date', inplace=True)
        
        return data

    @staticmethod
    def process_timestamp(data):
        """
        Process Timestamp Data
        """

        data["Year"] = data.index.year.astype(str)
        data["Month"] = data.index.month.astype(str)
        data.reset_index(inplace=True)
        data.drop(columns=["Date"], inplace=True)

        data["Month"] = data["Month"].apply(lambda x: "0"+ x if len(x)==1 else x)
        data["YM"] = data["Year"] + "-" + data["Month"]
        data["Date"] = pd.PeriodIndex(data["YM"], freq='M')
        
        data.set_index("Date", inplace=True)
        data.drop(columns=["Year", "Month", "YM"], inplace=True)

        return data

    def get_returns(self, data):
        """
        Compute Returns
        """
        data = data.pct_change()

        processed_data = self.process_timestamp(data)        
        
        return processed_data


class FamaFrenchFactorDataLoader:
    """
    Class to collect Time Series Data of Factors from Kenneth French's Library
    """

    def __init__(self, start=FF_START_DATE):
        self.start = start

    def get_factor_data(self, filenames):
        """
        Collects and Pre-Processes Factor Data
        """

        factor_data = pd.DataFrame()
        for filename in filenames:
            data = pdr.get_data_famafrench(filename, start=self.start)[0]
            data.reset_index(inplace=True)
            if not len(factor_data):
                factor_data = data
            else:
                factor_data = pd.merge(factor_data, data, how="inner", on="Date")
        
        factor_data.columns = [x.strip() for x in factor_data.columns.tolist()]
        factor_data.set_index("Date", inplace=True)
        
        return factor_data


def get_data(stock_tickers, market_ticker):
    """
    Function to collect Factor and Equity Data
    """

    print("-" * 50 + "Loading Time Series of Factors" + "-" * 50)
    famafrenchfactor = FamaFrenchFactorDataLoader()
    
    # Extract Fama-French (6) Factors data downloaded from library
    ff_data = famafrenchfactor.get_factor_data(filenames=FF_FILENAMES)
    ff_data = ff_data / 100

    print("-" * 50 + "Done!" + "-" * 50)

    print("-" * 50 + f"Loading Historical Prices of {len(STOCK_TICKERS)} Equities (Stocks)" + "-" * 50)
    # Extract Open/Close Prices of each Stock
    stock_data_obj = EquityDataLoader(tickers=stock_tickers)

    # Open Prices of Stocks (Monthly)
    stock_open_data = stock_data_obj.get_history(price_type="Open")
    processed_open_data = stock_data_obj.process_timestamp(stock_open_data)

    # Close Prices of Stocks (Monthly)
    stock_close_data = stock_data_obj.get_history(price_type="Close")
    processed_close_data = stock_data_obj.process_timestamp(stock_close_data)

    print("-" * 50 + "Done!" + "-" * 50)

    print("-" * 50 + f"Calculating Historical Returns of {len(STOCK_TICKERS)} Equities (Stocks)" + "-" * 50)
    # Monthly Returns on Individual Stocks
    stock_returns = stock_data_obj.get_returns(stock_close_data)
    print("-" * 50 + "Done!" + "-" * 50)

    print("-" * 50 + f"Loading Historical Prices of {MARKET_TICKER}" + "-" * 50)
    mkt_data_obj = EquityDataLoader(tickers=[market_ticker])
    spy_historical_data = mkt_data_obj.get_history(price_type="Close")
    spy_returns = mkt_data_obj.get_returns(spy_historical_data)

    print("-" * 50 + "Done!" + "-" * 50)


    return ff_data, processed_open_data, processed_close_data, stock_returns, spy_returns


class EarningsReportLoader:
    """
    Class to download Earnings Reports (Quarterly) from Yahoo Finance API
    """
    def __init__(self, tickers, from_file=True):
        self.tickers = tickers    
        self.from_file = from_file

    @staticmethod
    def save_earnings_file(earnings_response, filename):
        """
        serialize earnings report
        """
        
        with open(filename, 'wb') as f:
            pickle.dump(earnings_response, f)


    def get_earnings_response(self):
        """
        Etract earnings report from Alpha Vantage API and serialize file
        """
        # Holds earnings data for each ticker
        earnings = {}

        # Get earnings for each stock ticker
        for ticker in self.tickers:
            response = requests.get(f'{BASE_URL}function={FUNCTION}&symbol={ticker}&apikey={API_KEY}')
            earnings[ticker] = response.json()
        
        if self.from_file:
            self.save_earnings_file(earnings, "Earnings_Report.pkl")
        else:
            return earnings
    

    def quarterly_earnings_report(self, earnings_data, ticker):
        """
        Post-Process Earnings Report for a given equity
        """
        df_ticker = pd.DataFrame(earnings_data[ticker]['quarterlyEarnings'])
        
        # Convert reported date which is only applicable to quarterly earnings from string
        df_ticker['reportedDate'] = pd.to_datetime(df_ticker['reportedDate'])

        for field in EARNINGS_FIELDS:
            # non numeric are converted to NaN
            df_ticker[field] = pd.to_numeric(df_ticker[field], errors='coerce')
            
        # Add a column for the ticker
        df_ticker['ticker'] = ticker

        # Convert to dates which are in strings in raw format
        df_ticker['fiscalDateEnding'] = pd.to_datetime(df_ticker['fiscalDateEnding'])

        # # Sort by dates - we want the oldest date first
        df_ticker = df_ticker.sort_values('fiscalDateEnding')
        
        return df_ticker
    
    def get_earniings_history(self):
        """
        Generate Earnings History (Yearly)
        """
        
        # Create Response from AlphaVantage API 
        
        if self.from_file:
            with open("Earnings_Report.pkl", "rb") as f:
                earnings_data = pickle.load(f)
        else:
            earnings_data = self.get_earniings_history()

        # Aggregate Earnings Reports
        df_quarterly = pd.DataFrame()
        for ticker in self.tickers:
            df_quarterly = pd.concat([df_quarterly, self.quarterly_earnings_report(earnings_data=earnings_data, ticker=ticker)])

        df_quarterly["Year"] = df_quarterly["fiscalDateEnding"].dt.year.astype(str)
        df_quarterly1 = df_quarterly.groupby(["Year", "ticker"])[EARNINGS_FIELDS].mean().reset_index()
        df_quarterly1.set_index("Year", inplace=True)

        return df_quarterly1        


if __name__ == "__main__":

    ff_data, stock_open_data, stock_close_data, stock_returns, market_returns = get_data(stock_tickers=STOCK_TICKERS, market_ticker=MARKET_TICKER)
    
    print(ff_data.head())
    print(stock_open_data.head())
    print(stock_close_data.head())
    print(stock_returns.head())
    print(market_returns.head())