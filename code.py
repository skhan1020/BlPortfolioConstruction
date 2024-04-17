import numpy as np
import pandas_datareader as pdr
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from numpy.linalg import multi_dot
import matplotlib.pyplot as plt
from config import (
    STOCK_TICKERS,
    MARKET_TICKER,
    FF_FACTORS,
    RF_COL, 
    WINDOW, 
    ROLLING,
    INITIAL_CAPITAL,
    TOPN
)

class EquityDataLoader:
    def __init__(self, tickers, start="1963-07-01", end="2023-12-01"):
        self.tickers = tickers
        self.start = start
        self.end = end
    
    def get_history(self, price_type):
        
        data = pd.DataFrame()
        for ticker in self.tickers:
            yf_ticker = yf.Ticker(ticker)
            yf_ticker_history = yf_ticker.history(start=self.start, end=self.end, interval='1mo')
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

        data = data.pct_change()

        processed_data = self.process_timestamp(data)        
        
        return processed_data


class FamaFrenchFactorDataLoader:
    def __init__(self, start="1-1-1926"):
        self.start = start

    def get_factor_data(self, filenames):

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
    
    print("-" * 50 + "Loading Time Series of Factors" + "-" * 50)
    famafrenchfactor = FamaFrenchFactorDataLoader()
    
    # Extract Fama-French (6) Factors data downloaded from library
    ff_data = famafrenchfactor.get_factor_data(filenames=["F-F_Research_Data_5_Factors_2x3", "F-F_Momentum_Factor"])
    ff_data = ff_data / 100

    print("-" * 50 + f"Loading Historical Prices of {len(STOCK_TICKERS)} Equities (Stocks)" + "-" * 50)
    # Extract Open/Close Prices of each Stock
    stock_data_obj = EquityDataLoader(tickers=stock_tickers)

    # Open Prices of Stocks (Monthly)
    stock_open_data = stock_data_obj.get_history(price_type="Open")
    processed_open_data = stock_data_obj.process_timestamp(stock_open_data)

    # Close Prices of Stocks (Monthly)
    stock_close_data = stock_data_obj.get_history(price_type="Close")
    processed_close_data = stock_data_obj.process_timestamp(stock_close_data)

    # Monthly Returns on Individual Stocks
    stock_returns = stock_data_obj.get_returns(stock_close_data)

    print("-" * 50 + f"Loading Historical Prices of {MARKET_TICKER}" + "-" * 50)
    mkt_data_obj = EquityDataLoader(tickers=[market_ticker])
    spy_historical_data = mkt_data_obj.get_history(price_type="Close")
    spy_returns = mkt_data_obj.get_returns(spy_historical_data)

    return ff_data, processed_open_data, processed_close_data, stock_returns, spy_returns


class FamaFrenchModel:
    
    def __init__(self, stock, factors, rf_col, window=60, rolling=False):
        self.stock = stock
        self.rf_col = rf_col
        self.factors = factors
        self.window = window
        self.rolling = rolling
    
    def fit(self, ff_data, asset_data):
        
        print("-" * 50 + "Fitting Fama-French Factor Model" + "-" * 50)
        
        endog = asset_data[self.stock] - ff_data[self.rf_col]
        exog = sm.add_constant(ff_data[self.factors])
        if self.rolling:
            self.ff_model = RollingOLS(endog, exog, window=self.window)
        else:
            self.ff_model = sm.OLS(endog, exog)
        
        self.fitted_model = self.ff_model.fit()
        self.params = self.fitted_model.params

    def summary(self):
        print("-" * 50 + "Generating Summary" + "-" * 50)
        return self.fitted_model.summary()
    
    
    def partial_regression_plot(self):
        
        fig = plt.figure(figsize=(12, 8))
        sm.graphics.plot_partregress_grid(self.fitted_model, fig=fig)
        plt.savefig("Figures/Partial_Regression_Plots_" + self.stock + ".png")


    def rolling_beta_groups(self):

        rolling_betas = self.fitted_model.params.copy()
        rolling_betas.dropna(inplace=True)

        rolling_betas.rename(columns={"const": self.rf_col}, inplace=True)

        
        rolling_betas["Year"] = rolling_betas.index.year
        years = sorted(list(rolling_betas["Year"].unique()))
        year_ranges = [years[0]-1, years[len(years)//3], years[2*len(years)//3], years[-1]]
        labels = []
        for i in range(len(year_ranges)-1):
            labels.append(str(year_ranges[i]) + "-" + str(year_ranges[i+1]))

        rolling_betas["Group"] = pd.cut(rolling_betas["Year"], bins=year_ranges, labels=labels)

        return rolling_betas

    def mean_rolling_betas(self):
        
        rolling_betas = self.rolling_beta_groups()
        rolling_betas = rolling_betas.groupby("Group")[[self.rf_col] + self.factors].mean()
        
        return rolling_betas

    def plot_rolling_beta_groups(self):

        rolling_betas = self.rolling_beta_groups()
        
        rolling_betas.groupby(["Group"])[self.rf_col].hist(bins=100, density=True, legend=True)
        plt.ylabel(r"$\alpha_{}$".format({self.rf_col}))
        plt.title(r"{0} : Histogram Plot of $\alpha_{1}$ across different time intervals".format(self.stock,{self.rf_col}))
        plt.savefig(f"Figures/alpha_{self.rf_col}_histogram_{self.stock}.png")
        

        nrows, ncols=3, 2
        fig, axs  = plt.subplots(nrows=nrows, ncols=ncols, sharex='col', sharey='row')
        nrow = 0
        for i, factor in enumerate(self.factors):
            ncol = i % ncols
            if i % ncols == 0 and i > 0:
                nrow += 1

            rolling_betas.groupby(["Group"])[factor].hist(bins=100, density=True, ax=axs[nrow][ncol], figsize=(20, 10), legend=True)
            axs[nrow, ncol].set_ylabel(r"$\beta_{}$".format({factor}))
            axs[nrow, ncol].set_title(r"{0} : Histogram Plot of $\beta_{1}$ across different time intervals".format(self.stock,{factor}))

        fig.savefig(f"Figures/beta_{factor}_histogram_{self.stock}.png")

        plt.close()

    def plot_rolling_betas(self):
        
        print("-" * 50 + f"Rolling Estimates of Sensitivity Factors for {self.stock}" + "-" * 50)
        
        fig = self.fitted_model.plot_recursive_coefficient(variables=["const"] + self.factors, figsize=(14,6))
        plt.savefig(f"Rolling_Estimates_{self.stock}.png")                
        plt.close()
            


class FactorTradingStrategy:
    # def __init__(self, factor_df, open_df, close_df, initial_capital, topn):
    #ff_data, stock_open_data, stock_close_data, stock_returns, spy_returns
    def __init__(self, strategy, ff_data, open_df, close_df, stock_returns, market_returns, initial_capital=INITIAL_CAPITAL, topn=TOPN, market_ticker=MARKET_TICKER):

        # self.factor_df = factor_df.copy()
        #self.ff_data = ff_data.copy()
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
        
        # self.tickers = self.factor_df.columns.tolist()
        # self.factor_df.index = list(range(len(self.factor_df)))
        # self.open_df.index = list(range(len(self.open_df)))
        # self.close_df.index = list(range(len(self.close_df)))
        # self.trading_days = factor_df.index

    def holding_tickers(self, factor_data, topn):
        
        print("-" * 50 + "Rank Assets based on Factor" + "-" * 50)

        sorted_factors = factor_data.values.argsort(axis=1)
        factor_sorted_df = pd.DataFrame(sorted_factors)
        ticker_map = dict(zip(list(range(len(self.tickers))), self.tickers))
        for col in factor_sorted_df.columns:
            factor_sorted_df[col] = factor_sorted_df[col].map(ticker_map)

        factor_sorted_df.columns = ["Top " + str(i+1) for i in range(len(self.tickers))]

        return factor_sorted_df.iloc[:, :topn]


    def avg_rank_tickers_based_strategy(self):

        factor_data_list = list()
        for factor_trading_data in self.factor_trading_data_list:
            factor_data_list.append(self.holding_tickers(factor_data=factor_trading_data, topn= len(STOCK_TICKERS)))

        # for strategy in strategies:
        #     factor_data = get_factor(stocks=STOCK_TICKERS, factors=FF_FACTORS, stock_returns=self.stock_returns.copy(), ff_data=self.ff_data.copy(), factor=strategy)
        #     factor_data_list.append(self.holding_tickers(factor_data=factor_data, topn= len(STOCK_TICKERS)))

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
        
        return ranked_agg_factor_df



    def generate_signals(self):
        
        # factor_sorted_df = self.holding_tickers()
        factor_sorted_df = self.avg_rank_tickers_based_strategy()

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
        
        return signal_data
    

    @staticmethod
    def evalaute_strategy(signal_data, freq=12):
        
        cumulative_returns = signal_data["Portfolio Cumulative Returns"].tolist()[-1]
        annual_returns = signal_data["Portfolio Returns"].mean() * freq
        annual_volatility = signal_data["Portfolio Returns"].std() * np.sqrt(freq)
        sharpe_ratio = annual_returns / annual_volatility

        return pd.DataFrame({"Total Profit": [cumulative_returns], "Annual Returns": [annual_returns], "Annual Volatility": [annual_volatility], "Sharpe Ratio": [sharpe_ratio]})

    #@staticmethod
    # def plot_cumulative_returns(signal_data, market_data, market_ticker, factor):
    def plot_cumulative_returns(self, signal_data):

        strategy = '_'.join([x for x in self.strategy])
        print(f"Strategy : {strategy}")

        print("-" * 50 + f"Plot of Portfolio Cumulative Returns vs {self.market_ticker} Cumulative Returns for {strategy}" + "-" * 50)

        portfolio_cr = signal_data[["Portfolio Cumulative Returns"]]
        # market_data[f"{market_ticker} Cumulative Returns"] = (1 + market_data[market_ticker]).cumprod() - 1
        self.market_returns[f"{self.market_ticker} Cumulative Returns"] = (1 + self.market_returns[self.market_ticker]).cumprod() - 1
        

        # portfolio_market_returns = pd.merge(portfolio_cr, market_data, how="inner", on=portfolio_cr.index)
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
        

def get_factor(stocks, factors, stock_returns, ff_data, factor, rf_col=RF_COL, window=WINDOW, rolling=ROLLING):
        
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

    return factor_data


def determine_trading_days(factor_df, open_df, close_df, market_df):

    factor_start_index = factor_df.index[0]
    market_start_index = market_df.index[0]

    print("-" * 50 + "Determine Trading Days for Backtesting" + "-" * 50)

    if factor_start_index < market_start_index:
        factor_df = factor_df[factor_df.index.isin(market_df.index)]
        open_df = open_df[open_df.index.isin(market_df.index)]
        close_df = close_df[close_df.index.isin(market_df.index)]
    else:
        market_df = market_df[market_df.index.isin(factor_df.index)]
    
    return factor_df, open_df, close_df, market_df


def famafrench_regression_analysis():

    ff_data, _, _, stock_returns, _ = get_data(stock_tickers=STOCK_TICKERS, market_ticker=MARKET_TICKER)

    for stock in STOCK_TICKERS:
        model = FamaFrenchModel(stock=stock, factors=FF_FACTORS, rf_col=RF_COL, window=WINDOW, rolling=ROLLING)
        model.fit(asset_data=stock_returns, ff_data=ff_data)
        model.plot_rolling_beta_groups()
        model.plot_rolling_betas()


def backtest(strategies):

    ff_data, stock_open_data, stock_close_data, stock_returns, market_returns = get_data(stock_tickers=STOCK_TICKERS, market_ticker=MARKET_TICKER)

    performance_dict = dict()


    for strategy in strategies:
        factor_trading_strat = FactorTradingStrategy(strategy=strategy, ff_data=ff_data, open_df=stock_open_data, close_df=stock_close_data, stock_returns=stock_returns, market_returns=market_returns)
        signal_data = factor_trading_strat.generate_signals()

        factor_trading_strat.plot_cumulative_returns(signal_data=signal_data)

        performance = factor_trading_strat.evalaute_strategy(signal_data=signal_data)
        key = '_'.join([x for x in strategy])
        performance_dict.update({key: performance})
    
    return performance_dict

    # for factor in ["Alpha"] + FF_FACTORS:

        # factor_df = get_factor(stocks=STOCK_TICKERS, factors=FF_FACTORS, stock_returns=stock_returns, ff_data=ff_data, factor=factor)

        # factor_df1, stock_open_data1, stock_close_data1, spy_returns1 = determine_trading_days(factor_df=factor_df, open_df=stock_open_data, close_df=stock_close_data, market_df=spy_returns)

        # factor_trading_strat = FactorTradingStrategy(factor_df=factor_df1, open_df=stock_open_data1, close_df=stock_close_data1, initial_capital=INITIAL_CAPITAL, topn=TOPN)
        
        # signal_data = factor_trading_strat.generate_signals()

        # factor_trading_strat.plot_cumulative_returns(signal_data=signal_data, market_data=spy_returns1, market_ticker=MARKET_TICKER, factor=factor)

    #     performance = factor_trading_strat.evalaute_strategy(signal_data=signal_data)

    #     performance_dict.update({factor: performance})

    # return performance_dict


