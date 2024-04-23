from blportopt.config import (
    STOCK_TICKERS,
    MARKET_TICKER,
    FACTOR_COMBINATIONS,
)

from blportopt.data_utils import get_data
from blportopt.trading_strategy import (
    AlphaTradingStrategy,
    backtest
)

def calc_optimal_factors(metric):
    """
    Determine the optimal subset of factos that generate the optimal "Sharpe Ratio" or "Information Ratio"

    Parameters
    ----------

    metric : str
        Metric to determine Optimal Factor Set
    
    Returns
    -------

    optimal_factor_set: List[str]
        List of optimal factors based on Alpha Trading Strategy
    """

    all_performance = backtest(factor_combinations=FACTOR_COMBINATIONS, ALL=True)
   
    fc_dict = dict()
    for k, perf_df in all_performance.items():
        fc_dict.update({k: perf_df[metric].values[0]})
    
    optimal_factor_combination = max(fc_dict, key=fc_dict.get)

    optimal_factor_set = optimal_factor_combination.split("_")

    return optimal_factor_set


def portfolio_construction(optimal_factor_combination, n_assets):
    """
    Function to construct portfolio based on optimal factor subset 
    Naive weight allocation based on ranks of the Top 5 stocks

    Parameters
    ----------

    optimal_factor_combination : List[str]
        List of factors responsible for optimal performance of portfolio measured by SR/IR
    
    n_assets : int
        Number of assets to include in portfolio
    
    Returns
    -------

    List of assets to include in portfolio
    """

    ff_data, stock_open_data, stock_close_data, stock_returns, market_returns = get_data(stock_tickers=STOCK_TICKERS, market_ticker=MARKET_TICKER)

    alpha_trading_strat = AlphaTradingStrategy(factor_combination=optimal_factor_combination, ff_data=ff_data, open_df=stock_open_data, close_df=stock_close_data, stock_returns=stock_returns, market_returns=market_returns, topn=len(STOCK_TICKERS))
    avg_rank_assets = alpha_trading_strat.get_avg_rank()

    return avg_rank_assets[:n_assets]["Asset"].tolist()


if __name__ == "__main__":

    # Determine Optimal Set of Factor Loadings based on Sharpe Ratio of Portfolio
    optimal_factor_combination = calc_optimal_factors(metric="Sharpe Ratio")
    print(f"Optimal Factor Combination: {optimal_factor_combination}")

    # Performance Metrics of Optimal Set of Factor Loadings 
    optimal_performance = backtest(factor_combinations=[optimal_factor_combination])
    print(f"Optimal Performance Metrics : {optimal_performance}")

    # Optimal Portfolio Allocation
    optimal_portfolio = portfolio_construction(optimal_factor_combination=optimal_factor_combination, n_assets=5)
    print(f"Optimal Portfolio Allocation : {optimal_portfolio}")