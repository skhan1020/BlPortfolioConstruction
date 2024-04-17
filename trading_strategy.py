import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import (
    STOCK_TICKERS,
    MARKET_TICKER,
    FF_FACTORS,
    RF_COL, 
    WINDOW, 
    ROLLING,
    INITIAL_CAPITAL,
    TOPN,
    STRATEGIES,
)

from data_utils import get_data
from fama_french_model import FamaFrenchModel

def get_factor(stocks, factors, stock_returns, ff_data, factor, rf_col=RF_COL, window=WINDOW, rolling=ROLLING):
    """
    Extract Factor Values from Fama-French Model
    """
    if factor == "Alpha":
        factor = "const"

    factor_data = pd.DataFrame()
    for stock in stocks:
    
        print("-" * 50 + f"Calculate {factor} Coefficients for {stock} from Regression Analysis" + "-" * 50)

        model = FamaFrenchModel(stock=stock, factors=factors, rf_col=rf_col, window=window, rolling=rolling)
        model.fit(asset_data=stock_returns, ff_data=ff_data)
        factor_df = model.params[[factor]].dropna()
        factor_stock = pd.DataFrame(factor_df)
        factor_stock.columns = [stock]
        factor_data = pd.concat([factor_data, factor_stock], axis=1)

        print("-" * 50 + "Done!" + "-" * 50)

    return factor_data


def determine_trading_days(factor_df, open_df, close_df, market_df):
    """
    Compute Trading Days for stocks and SPY
    """
    factor_start_index = factor_df.index[0]
    market_start_index = market_df.index[0]

    print("-" * 50 + "Determine Trading Days for Backtesting" + "-" * 50)

    if factor_start_index < market_start_index:
        factor_df = factor_df[factor_df.index.isin(market_df.index)]
        open_df = open_df[open_df.index.isin(market_df.index)]
        close_df = close_df[close_df.index.isin(market_df.index)]
    else:
        market_df = market_df[market_df.index.isin(factor_df.index)]
    
    print("-" * 50 + "Done!" + "-" * 50)

    return factor_df, open_df, close_df, market_df



class FactorTradingStrategy:
    """
    Class to implement a trading strategy and compute P&L, Sharpe Ratios, and Information Ratios 
    by ranking ssets based on their factor values
    
    """
    def __init__(self, strategy, ff_data, open_df, close_df, stock_returns, market_returns, initial_capital=INITIAL_CAPITAL, topn=TOPN, market_ticker=MARKET_TICKER):

        self.strategy = strategy
        self.initial_capital = initial_capital
        self.topn = topn
        self.market_ticker = market_ticker

        self.factor_trading_data_list = list()
        for idx, factor in enumerate(self.strategy):
            factor_data = get_factor(stocks=STOCK_TICKERS, factors=FF_FACTORS, stock_returns=stock_returns.copy(deep=True), ff_data=ff_data.copy(deep=True), factor=factor)
            if idx == len(strategy) -1 :
                factor_trading_data, self.open_df, self.close_df, self.market_returns = determine_trading_days(factor_df=factor_data, open_df=open_df.copy(deep=True), close_df=close_df.copy(deep=True), market_df=market_returns.copy(deep=True))
            else:
                factor_trading_data, _, _, _ = determine_trading_days(factor_df=factor_data, open_df=open_df.copy(deep=True), close_df=close_df.copy(deep=True), market_df=market_returns.copy(deep=True))
            
            factor_trading_data.index = list(range(len(factor_trading_data)))
            self.factor_trading_data_list.append(factor_trading_data)

        self.tickers = self.open_df.columns.tolist()
        self.trading_days = self.open_df.index

        self.open_df.index = list(range(len(self.open_df)))
        self.close_df.index = list(range(len(self.close_df)))
        

    def holding_tickers(self, factor_data, topn):
        """
        Generate tickers to hold on a given trading day
        """
        print("-" * 50 + "Rank Assets" + "-" * 50)

        sorted_factors = factor_data.values.argsort(axis=1)
        factor_sorted_df = pd.DataFrame(sorted_factors)
        ticker_map = dict(zip(list(range(len(self.tickers))), self.tickers))
        for col in factor_sorted_df.columns:
            factor_sorted_df[col] = factor_sorted_df[col].map(ticker_map)

        factor_sorted_df.columns = ["Top " + str(i+1) for i in range(len(self.tickers))]
        
        print("-" * 50 + "Done!" + "-" * 50)

        return factor_sorted_df.iloc[:, :topn]


    def avg_rank_tickers_based_strategy(self):
        """
        Compute Average Rank of all equities based on the factors selected 
        """
        print("-" * 50 + f"Aggregate Ranking Based on Strategy : {'_'.join([x for x in self.strategy])}" + "-" * 50)

        factor_data_list = list()
        for factor_trading_data in self.factor_trading_data_list:
            factor_data_list.append(self.holding_tickers(factor_data=factor_trading_data, topn= len(STOCK_TICKERS)))


        def rank_tickers(df):

            df_list = df.to_dict('records')

            ranked_list = list()

            for dict_item in df_list:
                ranked_list.append({v: int(k.split()[1]) for k, v in dict_item.items()})
            
            ranked_factor_df = pd.DataFrame(ranked_list)
            
            return ranked_factor_df


        list_data = factor_data_list[0]
        dummy_list = [[0 for _ in range(len(list_data.columns))] for _ in range(len(list_data))]
        col_names = sorted(list_data.loc[0].values)
        rank_df = pd.DataFrame(dummy_list, columns=col_names)

        for factor_df in factor_data_list:
        
            ranked_factor_df = rank_tickers(factor_df)
            rank_df = rank_df.add(ranked_factor_df, fill_value=0)

        avg_rank_df = rank_df / len(factor_data_list)
        ranked_agg_factor_df = self.holding_tickers(factor_data=avg_rank_df, topn=self.topn)

        print("-" * 50 + "Done!" + "-" * 50)
        
        return ranked_agg_factor_df, avg_rank_df



    def generate_signals(self):
        """
        Portfolio Rebalancing - Compute P&L using a multi-factor based strategy
        """
        factor_sorted_df, _ = self.avg_rank_tickers_based_strategy()

        hold_assets, prev = list(), list()

        portfolio_data, signal_data = pd.DataFrame(), pd.DataFrame()

        print("-" * 50 + "Backtesting of Factors Based on Selected Strategy -- Compute Cumulatie Returns" + "-" * 50)

        for index, topn_assets in enumerate(factor_sorted_df.values):
            
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
        strategy = '_'.join([x for x in self.strategy])

        print("-" * 50 + f"Plot of Portfolio Cumulative Returns vs {self.market_ticker} Cumulative Returns for {strategy}" + "-" * 50)

        portfolio_cr = signal_data[["Portfolio Cumulative Returns"]]
        self.market_returns[f"{self.market_ticker} Cumulative Returns"] = (1 + self.market_returns[self.market_ticker]).cumprod() - 1
        

        portfolio_market_returns = pd.merge(portfolio_cr, self.market_returns, how="inner", on=portfolio_cr.index)
        portfolio_market_returns.dropna(inplace=True)
        portfolio_market_returns.drop(columns=[self.market_ticker], inplace=True)
        portfolio_market_returns.rename(columns={"key_0": "Date"}, inplace=True)
        portfolio_market_returns.set_index("Date", inplace=True)
        

        portfolio_market_returns.plot(kind='line', figsize=(10, 10),title=f"Backtesting {strategy} : Portfolio Cumulative Returns vs {self.market_ticker} Cumulative Returns")
        plt.xlabel('Date')
        plt.ylabel('Cumulative Returns')
        plt.grid(True)
        plt.savefig(f"Figures/Cumulative_Returns_{strategy}.png")
        plt.show()
        

def backtest(strategies, ALL=False):
    """
    Function to backtest Factor Based Strategy with Market Returns
    """
    ff_data, stock_open_data, stock_close_data, stock_returns, market_returns = get_data(stock_tickers=STOCK_TICKERS, market_ticker=MARKET_TICKER)

    performance_dict = dict()


    for strategy in strategies:
        factor_trading_strat = FactorTradingStrategy(strategy=strategy, ff_data=ff_data, open_df=stock_open_data, close_df=stock_close_data, stock_returns=stock_returns, market_returns=market_returns)
        signal_data = factor_trading_strat.generate_signals()

        if not ALL:
            factor_trading_strat.plot_cumulative_returns(signal_data=signal_data)

        performance = factor_trading_strat.evalaute_strategy(signal_data=signal_data)
        key = '_'.join([x for x in strategy])
        performance_dict.update({key: performance})


    return performance_dict



if __name__ == "__main__":


    # Backtest All Factors against S&P 500 Market Index Returns
    performance = backtest(strategies=STRATEGIES)
    print(performance)
