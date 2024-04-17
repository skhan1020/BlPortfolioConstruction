from config import (
    STOCK_TICKERS,
    MARKET_TICKER,
    ALL_STRATEGIES,
)

from data_utils import get_data
from trading_strategy import (
    FactorTradingStrategy,
    backtest
)

def calc_optimal_strategy(metric):
    """
    Determine the optimal subset of factos that generate the optimal "Sharpe Ratio" or "Information Ratio"
    """

    all_performance = backtest(strategies=ALL_STRATEGIES, ALL=True)
   
    strategy_dict = dict()
    for k, perf_df in all_performance.items():
        strategy_dict.update({k: perf_df[metric].values[0]})
    
    optimal_strategy = max(strategy_dict, key=strategy_dict.get)

    optimal_strategy1 = optimal_strategy.split("_")

    return optimal_strategy1


def portfolio_construction(optimal_strategy):
    """
    Function to construct portfolio based on optimal factor subset 
    Naive weight allocation based on ranks of the Top 5 stocks
    """
    ff_data, stock_open_data, stock_close_data, stock_returns, market_returns = get_data(stock_tickers=STOCK_TICKERS, market_ticker=MARKET_TICKER)

    factor_trading_strat = FactorTradingStrategy(strategy=optimal_strategy, ff_data=ff_data, open_df=stock_open_data, close_df=stock_close_data, stock_returns=stock_returns, market_returns=market_returns)

    _, avg_rank_factor_df = factor_trading_strat.avg_rank_tickers_based_strategy()

    asset_mean_rank_df = avg_rank_factor_df.mean().reset_index().sort_values(by=0).rename(columns={'index': 'Asset', 0: 'Rank'}).reset_index(drop=True)
    ranked_assets = dict(zip(asset_mean_rank_df.loc[:4, 'Asset'], 1/asset_mean_rank_df.loc[:4, 'Rank']))
    total_sum = sum(ranked_assets.values())
    portfolio = {k: v /total_sum for k, v in ranked_assets.items()}

    return portfolio



if __name__ == "__main__":

    # Determine Optimal Set of Factor Loadings based on Sharpe Ratio of Portfolio
    optimal_strategy = calc_optimal_strategy(metric="Sharpe Ratio")
    print(f"Optimal Strategy : {optimal_strategy}")

    # Performance Metrics of Optimal Set of Factor Loadings 
    optimal_performance = backtest(strategies=optimal_strategy)
    print(f"Optimal Performance Metrics : {optimal_performance}")

    # Optimal Portfolio Allocation
    optimal_portfolio = portfolio_construction(optimal_strategy=optimal_strategy)
    print(f"Optimal Portfolio Allocation : {optimal_portfolio}")