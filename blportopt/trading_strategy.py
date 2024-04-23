import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from blportopt.config import (
    STOCK_TICKERS,
    MARKET_TICKER,
    FACTOR_COMBINATIONS,
    RF_COL, 
    WINDOW, 
    ROLLING,
    INITIAL_CAPITAL,
    TOPN,
    FIGURES_DIR,
)

from blportopt.data_utils import get_data
from blportopt.fama_french_model import FamaFrenchModel

def get_alpha(stocks, factors, stock_returns, ff_data, rf_col=RF_COL, window=WINDOW, rolling=ROLLING):
    """
    Extract Alpha Values from Fama-French Model

    Parameters
    ----------

    stocks : List[str]
        List of stock ticker symbols
    
    factors : List[str]
        List of factors included in Fama-French Model
    
    stock_returns : pd.DataFrame
        Historical returns of stocks
    
    ff_data : pd.DataFrame
        Fama-French time series data 
    
    rf_col : str
        Risk-free rate column name
    
    window : int
        Size of window for rolling regression
    
    rolling : boolean
        Flag for rolling regression
    
    Returns
    -------

    alpha_data : pd.DataFrame
        Alpha values of all stocks from Fama-French model

    """


    alpha_data = pd.DataFrame()
    for stock in stocks:
    
        print("-" * 50 + f"Calculate Alpha : {stock} (Fama French Model)" + "-" * 50)

        model = FamaFrenchModel(stock=stock, factors=factors, rf_col=rf_col, window=window, rolling=rolling)
        model.fit(asset_data=stock_returns, ff_data=ff_data)
        alpha_df = model.params[["const"]].dropna()
        alpha_stock = pd.DataFrame(alpha_df)
        alpha_stock.columns = [stock]
        alpha_data = pd.concat([alpha_data, alpha_stock], axis=1)

        print("-" * 50 + "Done!" + "-" * 50)

    return alpha_data


def determine_trading_days(alpha_df, open_df, close_df, market_df):
    """
    Compute Trading Days for stocks and SPY

    Parameters
    ----------

    alpha_df : pd.DataFrame
        Alpha values obtained from Fama-French model
    
    open_df : pd.DataFrame
        Historical open stock prices 
    
    close_df : pd.DataFrame
        Historical closing stock prices
    
    market_df : pd.DataFrame
        Historical SPY returns
    
    Returns
    -------

    alpha_df : pd.DataFrame
        Alpha values on Trading Days
    
    open_df : pd.DataFrame
        Open stock prices on Trading Days
    
    close_df : pd.DataFrame
        Close stock prices on Trading Days
    
    market_df : pd.DataFrame
        SPY returns on Trading Days
    """
    alpha_start_index = alpha_df.index[0]
    market_start_index = market_df.index[0]

    print("-" * 50 + "Determine Trading Days for Backtesting" + "-" * 50)

    if alpha_start_index < market_start_index:
        alpha_df = alpha_df[alpha_df.index.isin(market_df.index)]
        open_df = open_df[open_df.index.isin(market_df.index)]
        close_df = close_df[close_df.index.isin(market_df.index)]
    else:
        market_df = market_df[market_df.index.isin(alpha_df.index)]
    
    print("-" * 50 + "Done!" + "-" * 50)

    return alpha_df, open_df, close_df, market_df



class AlphaTradingStrategy:
    """
    Class to implement Alpha Trading Strategy and compute P&L, Sharpe Ratios, and Information Ratios 
    by ranking assets based on their alpha coefficients
    
    Parameters
    ----------

    factor_combination : List[str]
        List of factors used in Fama-French model
    
    ff_data : pd.DataFrame
        Time series of Fama-French data

    open_df : pd.DataFrame
        Historical open stock prices 
        
    close_df : pd.DataFrame
        Historical closing stock prices
    
    stock_returns : pd.DataFrame
        Historical stock returns

    market_returns : pd.DataFrame
        Historical SPY/Benchmark market index returns
    
    initial_capital : int
        Initial Capital used to backtest strategy
    
    topn : int
        Top N stocks based on Alpha values
    
    market_ticker : str
        Benchmark market index ticker (eg SPY)

    """
    def __init__(self, factor_combination, ff_data, open_df, close_df, stock_returns, market_returns, initial_capital=INITIAL_CAPITAL, topn=TOPN, market_ticker=MARKET_TICKER):

        self.factor_combination = factor_combination
        self.initial_capital = initial_capital
        self.topn = topn
        self.market_ticker = market_ticker

        #self.alpha_trading_data_list = list()

        alpha_df = get_alpha(stocks=STOCK_TICKERS, factors=self.factor_combination, stock_returns=stock_returns.copy(deep=True), ff_data=ff_data.copy(deep=True))
        self.alpha_trading_data, self.open_df, self.close_df, self.market_returns = determine_trading_days(alpha_df=alpha_df, open_df=open_df.copy(deep=True), close_df=close_df.copy(deep=True), market_df=market_returns.copy(deep=True))

        self.alpha_trading_data.index = list(range(len(self.alpha_trading_data)))
        #self.alpha_trading_data_list.append(alpha_trading_data)

        self.tickers = self.open_df.columns.tolist()
        self.trading_days = self.open_df.index

        self.open_df.index = list(range(len(self.open_df)))
        self.close_df.index = list(range(len(self.close_df)))
        

    def holding_tickers(self):
        """
        Generate tickers to buy & hold on a given trading day based on alpha values

        Returns
        -------

        alpha_sorted_df : pd.DataFrame
            pandas dataframe of alpha values of stocks in portfolio
        """

        print("-" * 50 + "Rank Assets" + "-" * 50)

        sorted_alpha = self.alpha_trading_data.values.argsort(axis=1)
        alpha_sorted_df = pd.DataFrame(sorted_alpha)
        ticker_map = dict(zip(list(range(len(self.tickers))), self.tickers))
        for col in alpha_sorted_df.columns:
            alpha_sorted_df[col] = alpha_sorted_df[col].map(ticker_map)

        alpha_sorted_df.columns = ["Top " + str(i+1) for i in range(len(self.tickers))]
        
        print("-" * 50 + "Done!" + "-" * 50)

        return alpha_sorted_df.iloc[:, :self.topn]



    def get_avg_rank(self):
        """
        Average Rank of Assets based on their Alpha Values on each trading day

        Returns
        -------

        asset_mean_rank_df : pd.DataFrame
            average rank of each asset in ascending order
        """
        alpha_sorted_df = self.holding_tickers()

        def rank_tickers(df):

            df_list = df.to_dict('records')

            ranked_list = list()

            for dict_item in df_list:
                ranked_list.append({v: int(k.split()[1]) for k, v in dict_item.items()})
            
            ranked_factor_df = pd.DataFrame(ranked_list)
            
            return ranked_factor_df

        ranked_alpha_df = rank_tickers(df=alpha_sorted_df)
        
        asset_mean_rank_df = ranked_alpha_df.mean().reset_index().sort_values(by=0).rename(columns={'index': 'Asset', 0: 'Rank'}).reset_index(drop=True)
        
        return asset_mean_rank_df
    


    def generate_signals(self):
        """
        Portfolio Rebalancing - Compute P&L using a multi-factor based strategy

        Returns
        -------

        signal_data : pd.DataFrame
            Portfolio holdings based on alpha trading strategy with Returns
        """

        alpha_sorted_df = self.holding_tickers()

        hold_assets, prev = list(), list()

        portfolio_data, signal_data = pd.DataFrame(), pd.DataFrame()

        print("-" * 50 + "Backtesting of Factors Based on Selected Strategy -- Compute Cumulatie Returns" + "-" * 50)

        for index, topn_assets in enumerate(alpha_sorted_df.values):
            
            portfolio_holdings = self.close_df.loc[index, topn_assets].sum()
            portfolio_df = pd.DataFrame({"Portfolio Holdings": portfolio_holdings}, index=[index])
            portfolio_data = pd.concat([portfolio_data, portfolio_df])

            hold_assets = list(set(prev) & set(topn_assets))
    
            if len(hold_assets):
                buy_assets = list(set(topn_assets) - set(hold_assets))
            else:
                buy_assets = topn_assets

            buy = self.open_df.loc[index, buy_assets].sum()

            
            sell_assets = list(set(prev) - set(topn_assets))

            if len(sell_assets):
                sell = self.open_df.loc[index, sell_assets].sum()
            else:
                sell = 0

            prev = topn_assets

            signal_df = pd.DataFrame({"Buy": [buy], "Sell": [sell]}, index=[index])
            signal_data = pd.concat([signal_data, signal_df])

        
        signal_data["Portfolio Holdings"] = portfolio_data["Portfolio Holdings"].cumsum()
        signal_data["Cash"] = self.initial_capital  - (signal_data["Buy"] - signal_data["Sell"]).cumsum()
        signal_data["Total Portfolio"] = signal_data["Cash"] + signal_data["Portfolio Holdings"]
        signal_data["Portfolio Returns"] = signal_data["Total Portfolio"].pct_change()
        signal_data["Portfolio Cumulative Returns"] = (1 + signal_data["Portfolio Returns"]).cumprod() - 1

        signal_data.index = self.trading_days
        
        print("-" * 50 + "Done!" + "-" * 50)

        return signal_data
    

    def evalaute_strategy(self, signal_data, freq=12):
        """
        Measure Performance of the Portfolio - 
            i. Cumulative Returns
            ii. Annual Returns
            iii. Annual Volatilities
            iv. Sharpe Ratio
            v. Information Ratio
        
        Parameters
        ----------
        
        signal_data : pd.DataFrame
            Portfolio holdings based on alpha trading strategy with Returns

        freq : int
            Time period (monthly, annualised, etc) of returns

        Returns
        -------
        
        pandas dataframe with net profit, annual returns, annual volatility, sharpe ratio, and information ratio
        """
        cumulative_returns = signal_data["Portfolio Cumulative Returns"].tolist()[-1]
        annual_returns = signal_data["Portfolio Returns"].mean() * freq
        annual_volatility = signal_data["Portfolio Returns"].std() * np.sqrt(freq)
        sharpe_ratio = annual_returns / annual_volatility
        market_returns = self.market_returns[self.market_ticker].mean() * freq
        diff = signal_data["Portfolio Returns"] - self.market_returns[self.market_ticker]
        relative_std = diff.std() * np.sqrt(freq)
        information_ratio = (annual_returns - market_returns) / relative_std


        return pd.DataFrame({"Total Profit": [cumulative_returns], "Annual Returns": [annual_returns], "Annual Volatility": [annual_volatility], "Sharpe Ratio": [sharpe_ratio], "Information Ratio": [information_ratio]})

    def plot_cumulative_returns(self, signal_data):
        """
        Plot of Cumulative Returns generated by Multi-Factor trading strategy and Market (SPY) returns
        """

        factor_combination = '_'.join([x for x in self.factor_combination])

        print("-" * 50 + f"Plot of Portfolio Cumulative Returns vs {self.market_ticker} Cumulative Returns for {factor_combination}" + "-" * 50)

        portfolio_cr = signal_data[["Portfolio Cumulative Returns"]]
        self.market_returns[f"{self.market_ticker} Cumulative Returns"] = (1 + self.market_returns[self.market_ticker]).cumprod() - 1
        

        portfolio_market_returns = pd.merge(portfolio_cr, self.market_returns, how="inner", on=portfolio_cr.index)
        portfolio_market_returns.dropna(inplace=True)
        portfolio_market_returns.drop(columns=[self.market_ticker], inplace=True)
        portfolio_market_returns.rename(columns={"key_0": "Date"}, inplace=True)
        portfolio_market_returns.set_index("Date", inplace=True)
        

        portfolio_market_returns.plot(kind='line', figsize=(10, 10),title=f"Backtesting {factor_combination} : Portfolio Cumulative Returns vs {self.market_ticker} Cumulative Returns")
        plt.xlabel('Date')
        plt.ylabel('Cumulative Returns')
        plt.grid(True)
        plt.savefig(os.path.join(FIGURES_DIR, f"Cumulative_Returns_{factor_combination}.png"))
        plt.show()
        

def backtest(factor_combinations, ALL=False):
    """
    Function to backtest Factor Based Strategy with Market Returns
    
    Parameters
    ----------

    factor_combinations : List[str]
        List of factors used in the Fama-French Model to evaluate Alpha Trading Strategy

    ALL : boolean
        Boolean flag to backtest all possible factor combinations
    
    Returns
    -------

    performance_dict : Dict[str, pd.DataFrame]
        Dictionary of factor combination and performance metrics
    """

    ff_data, stock_open_data, stock_close_data, stock_returns, market_returns = get_data(stock_tickers=STOCK_TICKERS, market_ticker=MARKET_TICKER)

    performance_dict = dict()


    for factor_combination in factor_combinations:
        alpha_trading_strat = AlphaTradingStrategy(factor_combination=factor_combination, ff_data=ff_data, open_df=stock_open_data, close_df=stock_close_data, stock_returns=stock_returns, market_returns=market_returns)
        signal_data = alpha_trading_strat.generate_signals()

        if not ALL:
            alpha_trading_strat.plot_cumulative_returns(signal_data=signal_data)

        performance = alpha_trading_strat.evalaute_strategy(signal_data=signal_data)
        key = '_'.join([x for x in factor_combination])
        performance_dict.update({key: performance})


    return performance_dict



if __name__ == "__main__":


    # Backtest All Factors against S&P 500 Market Index Returns
    performance = backtest(factor_combinations=FACTOR_COMBINATIONS)
    print(performance)
