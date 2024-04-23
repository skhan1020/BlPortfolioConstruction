import os
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
import matplotlib.pyplot as plt
from blportopt.config import (
    STOCK_TICKERS,
    MARKET_TICKER,
    FACTOR_COMBINATIONS,
    RF_COL, 
    WINDOW, 
    ROLLING,
    FIGURES_DIR,
)
from blportopt.data_utils import get_data

class FamaFrenchModel:
    """
    Fama French Factor Model using Rolling Regression

    Parameters
    ----------

    stock : str
        Stock Ticker Symbol

    factors : List[str]
        List of Fama-French Factors 

    rf_col : str
        risk-free rate column name from fama-french dataset
    
    window : int
        Rolling window size for regression analysis
    
    rolling: boolean
        Flag to perform rolling regression

    """
    def __init__(self, stock, factors, rf_col, window=60, rolling=False):
        self.stock = stock
        self.rf_col = rf_col
        self.factors = factors
        self.factor_combinations = "_".join([x for x in factors])
        self.window = window
        self.rolling = rolling
    
    def fit(self, ff_data, asset_data):
        """
        Fitting the Fama-French model (Rolling Estimation and const. Coefficient Estimation)

        parameters
        ----------

        ff_data : pd.DataFrame
            Fama-French Dataset with time series of Factors used in the model

        asset_data: pd.DataFrame
            Historical Returns of a particular asset
            
        """
        print("-" * 50 + "Fitting Fama-French Factor Model" + "-" * 50)
        
        endog = asset_data[self.stock] - ff_data[self.rf_col]
        exog = sm.add_constant(ff_data[self.factors])
        if self.rolling:
            self.ff_model = RollingOLS(endog, exog, window=self.window)
        else:
            self.ff_model = sm.OLS(endog, exog)
        
        self.fitted_model = self.ff_model.fit()
        
        print("-" * 50 + "Done!" + "-" * 50)

        self.params = self.fitted_model.params


    def summary(self):
        """
        Summary statistics of Fitted Model

        Returns
        -------
            Summary of fitted model
        """

        print("-" * 50 + "Generating Summary" + "-" * 50)
        return self.fitted_model.summary()
    
    
    def partial_regression_plot(self):
        """
        Partial Regression Plots of Each Factor
        """

        fig = plt.figure(figsize=(12, 8))
        sm.graphics.plot_partregress_grid(self.fitted_model, fig=fig)
        plt.savefig(os.path.join(FIGURES_DIR, "Partial_Regression_Plots_" + self.stock + ".png"))


    def rolling_beta_groups(self):
        """
        Grouping Rolling Betas into Distinct Time Intervals

        Returns
        --------
        rolling_betas : pd.DataFrame
            rolling beta estimates of factors grouped into time intervals
        """

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
        """
        Compute Mean of Rolling Betas within each Grooup
        """

        rolling_betas = self.rolling_beta_groups()
        rolling_betas = rolling_betas.groupby("Group")[[self.rf_col] + self.factors].mean()
        
        return rolling_betas

    def plot_rolling_beta_groups(self):
        """
        Historgram of Factors across different time intervals
        
        """
        rolling_betas = self.rolling_beta_groups()
        
        rolling_betas.groupby(["Group"])[self.rf_col].hist(bins=100, density=True, legend=True)
        plt.ylabel(r"$\alpha_{}$".format({self.rf_col}))
        plt.title(r"{0} : $\alpha_{1}$".format(self.stock,{self.rf_col}))
        plt.savefig(os.path.join(FIGURES_DIR, f"Alpha_{self.rf_col}_Histogram_{self.stock}_{self.factor_combinations}.png"))
        plt.close()

        if len(self.factors) == 1:
            nrows, ncols = 1, 1
        else:
            nrows, ncols=3, 2
        fig, axs  = plt.subplots(nrows=nrows, ncols=ncols, sharex='col', sharey='row')
        nrow = 0
        for i, factor in enumerate(self.factors):
            ncol = i % ncols
            if i % ncols == 0 and i > 0:
                nrow += 1
            if len(self.factors) == 1:
                ax = axs
            else:
                ax = axs[nrow, ncol]

            rolling_betas.groupby(["Group"])[factor].hist(bins=100, density=True, ax=ax, figsize=(10, 10), legend=True)
            ax.set_ylabel(r"$\beta_{}$".format({factor}))
            ax.set_title(r"{0} : $\beta_{1}$".format(self.stock,{factor}))

        fig.savefig(os.path.join(FIGURES_DIR, f"Factor_Histogram_{self.stock}_{self.factor_combinations}.png"))

        plt.close()

    def plot_rolling_betas(self):
        """
        Rolling Estimate Plots of Factors
        """

        print("-" * 50 + f"Rolling Estimates of Sensitivity Factors for {self.stock}" + "-" * 50)
        
        fig = self.fitted_model.plot_recursive_coefficient(variables=["const"] + self.factors, figsize=(20,30))
        plt.savefig(os.path.join(FIGURES_DIR, f"Rolling_Estimates_{self.stock}_{self.factor_combinations}.png"))                
        plt.close()

        print("-" * 50 + "Done!" + "-" * 50)


def famafrench_regression_analysis():
    """
    Function to perform Regression Analysis using Fama-French Model for 
    each stock and plots of rolling estimates and histograms
    """
    ff_data, _, _, stock_returns, _ = get_data(stock_tickers=STOCK_TICKERS, market_ticker=MARKET_TICKER)

    for factors in FACTOR_COMBINATIONS:
        for stock in STOCK_TICKERS:
            model = FamaFrenchModel(stock=stock, factors=factors, rf_col=RF_COL, window=WINDOW, rolling=ROLLING)
            model.fit(asset_data=stock_returns, ff_data=ff_data)
            model.plot_rolling_beta_groups()
            model.plot_rolling_betas()


if __name__ == "__main__":


    # Perform Fama-French Regression Analysis (Single & 6 Factor FF Model)
    famafrench_regression_analysis()