import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from blportopt.config import (
    # STOCK_TICKERS,
    ASSET_TICKERS,
    MARKET_TICKER,
    FF_FACTORS,
    RF_COL, 
    WINDOW, 
    ROLLING,
    INITIAL_CAPITAL,
    TOPN,
    FIGURES_DIR,
)
from blportopt.data_utils import (
    get_data,
    MarketCapEvaluator,
)
from blportopt.fama_french_model import (
    FFModelConfig,
    FamaFrenchModel,
)

def get_alpha(asset_type, factors, asset_returns, ff_data, rf_col=RF_COL, window=WINDOW, rolling=ROLLING):
    """
    Extract Alpha Values from Fama-French Model

    Parameters
    ----------

    asset_type : str
        Type of Equities (stocks/funds)
    
    factors : List[str]
        List of factors included in Fama-French Model
    
    asset_returns : pd.DataFrame
        Historical returns of assets
    
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

    ff_model_config = FFModelConfig(rf_col=rf_col, window=window, rolling=rolling)
    ff_model_config.factors = factors

    for asset in ASSET_TICKERS[asset_type]:
    
        print("-" * 50 + f"Calculate Alpha : {asset} (Fama French Model)" + "-" * 50)

        model = FamaFrenchModel(asset=asset, model_config=ff_model_config)
        model.fit(asset_data=asset_returns, ff_data=ff_data)
        alpha_df = model.params[["const"]].dropna()
        alpha_stock = pd.DataFrame(alpha_df)
        alpha_stock.columns = [asset]
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
        open_df = open_df[open_df.index.isin(alpha_df.index)]
        close_df = close_df[close_df.index.isin(alpha_df.index)]
    
    print("-" * 50 + "Done!" + "-" * 50)

    return alpha_df, open_df, close_df, market_df



class AlphaTradingStrategy:
    """
    Class to implement Alpha Trading Strategy and compute P&L, Sharpe Ratios, and Information Ratios 
    by ranking assets based on their alpha coefficients
    
    Parameters
    ----------

    asset_type : str
        Equity type (stocks/funds)

    allocations : Dict[str, float]
        Dictionary of asset allocations {Asset : Allocation}

    factors : List[str]
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
    def __init__(self, asset_type, allocations, factors, ff_data, open_df, close_df, asset_returns, market_returns, initial_capital=INITIAL_CAPITAL, topn=TOPN, market_ticker=MARKET_TICKER):

        self.factors = factors
        self.initial_capital = initial_capital
        self.topn = topn
        self.market_ticker = market_ticker

        alpha_df = get_alpha(asset_type=asset_type, factors=factors, asset_returns=asset_returns.copy(deep=True), ff_data=ff_data.copy(deep=True))

        self.alpha_trading_data, self.open_df, self.close_df, self.market_returns = determine_trading_days(alpha_df=alpha_df, open_df=open_df.copy(deep=True), close_df=close_df.copy(deep=True), market_df=market_returns.copy(deep=True))

        self.alpha_trading_data.index = list(range(len(self.alpha_trading_data)))

        self.tickers = self.open_df.columns.tolist()
        self.trading_days = self.open_df.index

        self.open_df.index = list(range(len(self.open_df)))
        self.close_df.index = list(range(len(self.close_df)))
        
        for asset, wt in allocations.items():
            self.close_df[asset], self.open_df[asset] = self.close_df[asset]*wt, self.open_df[asset]*wt

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
        

def plot_cumulative_returns(signal_data_combined):
    """
    Plot of Cumulative Returns generated from multiple asset allocations : Market Cap, Empirical Covariance Estimate, Covariance from Multi-Factor Model

    signal_data_combined : pd.DataFrame
        Cumulative Returns generated from multiple asset allocations : Market Cap, Empirical Covariance Estimate, Covariance from Multi-Factor Model
        
    """


    print("-" * 50 + "Plot of Portfolio Cumulative Returns" + "-" * 50)

    signal_data_combined.plot(kind='line', figsize=(10, 10))
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.title("MSR Performance : Portfolio Cumulative Returns")
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(os.path.join(FIGURES_DIR, f"MSR_Performance_Portfolios.png"))
    plt.show()

def backtest(allocations_dict, factors, asset_type):
    """
    Function to backtest Factor Based Strategy with Market Returns
    
    Parameters
    ----------

    allocation_dict : Dict[str, Dict[str, float]]
        Dictionary of asset allocations computed using different methods -- Market Cap, Empirical Covariance Matrix, Fama-French Covariance Matrix

    factors : List[str]
        List of factors used in the Fama-French Model to evaluate Alpha Trading Strategy

    asset_type : str
        Equity type (stocks/funds)
        
    Returns
    -------

    performance_dict : Dict[str, pd.DataFrame]
        Dictionary of factor combination and performance metrics
    """

    ff_data, asset_open_data, asset_close_data, asset_returns, market_returns = get_data(asset_tickers=ASSET_TICKERS[asset_type], market_ticker=MARKET_TICKER, asset_type=asset_type)

    performance_dict, signal_data_df = dict(), pd.DataFrame()

    for method, allocations in allocations_dict.items():

        alpha_trading_strat = AlphaTradingStrategy(asset_type=asset_type, allocations=allocations, factors=factors, ff_data=ff_data, open_df=asset_open_data, close_df=asset_close_data, asset_returns=asset_returns, market_returns=market_returns)
        
        signal_data = alpha_trading_strat.generate_signals()
        signal_data_df = pd.concat([signal_data_df, signal_data[["Portfolio Cumulative Returns"]]], axis=1)
        signal_data_df.rename(columns={"Portfolio Cumulative Returns": method}, inplace=True)    
        performance = alpha_trading_strat.evalaute_strategy(signal_data=signal_data)
        performance_dict.update({method: performance})

    plot_cumulative_returns(signal_data_combined=signal_data_df)


    return performance_dict



if __name__ == "__main__":

    asset_type = "stock"
    
    # Evaluate Market Capitalization of Assets
    mcap_eval = MarketCapEvaluator(tickers=ASSET_TICKERS[asset_type])
    market_cap_stocks = mcap_eval.compute_market_cap()

    # Allocations Dictionary
    allocations_dict = {
        "Market Cap": market_cap_stocks,
    }

    # Estimate MSR Portfolio Performance from different asset allocations
    performance = backtest(allocations_dict, factors=FF_FACTORS, asset_type=asset_type)
    print(performance)
