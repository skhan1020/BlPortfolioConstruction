import os
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
import matplotlib.pyplot as plt
from blportopt.config import (
    ASSET_TICKERS,
    MARKET_TICKER,
    FF_FACTORS,
    FACTOR_COMBINATIONS,
    RF_COL, 
    WINDOW, 
    ROLLING,
    FIGURES_DIR,
)
from blportopt.data_utils import get_data


class FFModelConfig:

    factors = FF_FACTORS
    rf_col = RF_COL
    window = WINDOW
    rolling =ROLLING

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class FamaFrenchModel:
    """
    Fama French Factor Model using Rolling Regression

    Parameters
    ----------

    stock : str
        Stock Ticker Symbol

    factors : List[str]
        List of Fama-French Factors 
    
    model_config : FFModelConfig
        Object to configure Fama-French Factor Model hyperparameters
    """
    def __init__(self, asset, model_config):
        self.asset = asset
        self.factors = model_config.factors
        #self.factors = factors
        self.factor_combinations = "_".join([x for x in self.factors])
        self.rf_col = model_config.rf_col
        self.window = model_config.window
        self.rolling = model_config.rolling
    
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

        asset_ff_data = pd.merge(asset_data[self.asset], ff_data, how="inner", on="Date")
        asset_ff_data.dropna(inplace=True)

        endog = asset_ff_data[self.asset] - asset_ff_data[self.rf_col]
        exog = sm.add_constant(asset_ff_data[self.factors])

        if self.rolling:
            self.ff_model = RollingOLS(endog, exog, window=self.window)
        else:
            self.ff_model = sm.OLS(endog, exog)
        
        self.fitted_model = self.ff_model.fit()
        
        print("-" * 50 + "Done!" + "-" * 50)

        self.params = self.fitted_model.params
        self.X = exog
        self.y = endog

    def rolling_residuals(self):
        """
        Calculate rolling residuals of fitted model

        Returns
        -------
            Rolling Residuals of Fama-French Factor Model
        """

        resid = pd.DataFrame(self.y - (self.params * self.X).sum(1), columns=[self.asset])

        return resid
    
    def rsquared_adj(self):

        return self.fitted_model.rsquared_adj.mean()

    def plot_rolling_residuals(self):

        residuals = self.rolling_residuals()
        residuals.plot()
        plt.ylabel("Residuals")
        plt.savefig(os.path.join(FIGURES_DIR, f"Residuals_{self.asset}_{self.factor_combinations}.png"))
        plt.close()

    def plot_rolling_correlations_residuals_vs_factor(self, factor_set):

        residuals = self.rolling_residuals()
        
        if "Mkt-RF" in set(factor_set):
            factor_val = "Mkt-RF"
        else:
            factor_val = factor_set[0]
        factors = self.X.loc[:, [factor_val]]

        df_resid_beta = pd.concat([residuals, factors], axis=1)
        df_resid_beta_corr = df_resid_beta.rolling(self.window).corr(pairwise=True).reset_index()

        df_resid_beta_corr = df_resid_beta_corr[df_resid_beta_corr["level_1"]==factor_val][["Date",self.asset]].dropna()
        df_resid_beta_corr.set_index("Date", inplace=True)

        df_resid_beta_corr.plot(figsize=(8, 8))
        plt.xlabel(r"$\beta_{}$".format({factor_val}))
        plt.ylabel(f"$\sigma$")
        plt.title(r"Correlation of Residuals vs $\beta_{0}$".format({factor_val}))
        plt.savefig(os.path.join(FIGURES_DIR, f"Correlation_Residuals_{self.asset}_{self.factor_combinations}.png"))
        plt.close()

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
        plt.savefig(os.path.join(FIGURES_DIR, f"Partial_Regression_Plots_{self.asset}_{self.factor_combinations}.png"))


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
        plt.title(r"{0} : $\alpha_{1}$".format(self.asset,{self.rf_col}))
        plt.savefig(os.path.join(FIGURES_DIR, f"Alpha_{self.rf_col}_Histogram_{self.asset}_{self.factor_combinations}.png"))
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
            ax.set_title(r"{0} : $\beta_{1}$".format(self.asset,{factor}))

        fig.savefig(os.path.join(FIGURES_DIR, f"Factor_Histogram_{self.asset}_{self.factor_combinations}.png"))

        plt.close()

    def plot_rolling_betas(self):
        """
        Rolling Estimate Plots of Factors
        """

        print("-" * 50 + f"Rolling Estimates of Sensitivity Factors for {self.asset}" + "-" * 50)
        
        fig = self.fitted_model.plot_recursive_coefficient(variables=["const"] + self.factors, figsize=(20,30))
        plt.savefig(os.path.join(FIGURES_DIR, f"Rolling_Estimates_{self.asset}_{self.factor_combinations}.png"))                
        plt.close()

        print("-" * 50 + "Done!" + "-" * 50)


def famafrench_regression_analysis(asset_type, factor_combinations=FACTOR_COMBINATIONS, rf_col=RF_COL, window=WINDOW, rolling=ROLLING):
    """
    Function to perform Regression Analysis using Fama-French Model for 
    each asset 
    
    a) Plots rolling estimates of residuals for each asset-factor combination
    b) Plots rolling estimates of correlationss between residuals and factors
    c) Plots rolling estimates of factors
    d) Plots histograms of factors grouped into three different time periods
    """

    ff_data, _, _, asset_returns, _ = get_data(asset_tickers=ASSET_TICKERS[asset_type], market_ticker=MARKET_TICKER, asset_type=asset_type)

    ff_model_config = FFModelConfig(rf_col=rf_col, window=window, rolling=rolling)

    ff_asset_rsq, ff_asset_params = pd.DataFrame(), pd.DataFrame()
    for factors in factor_combinations:
        df_asset_rsq, df_asset_params = pd.DataFrame(), pd.DataFrame()
        for asset in ASSET_TICKERS[asset_type]:
            
            ff_model_config.factors = factors

            # Instantiate the Fama-French Factor model
            model = FamaFrenchModel(asset=asset, model_config=ff_model_config)
            
            # Fitting of model
            model.fit(asset_data=asset_returns, ff_data=ff_data)
            
            if rolling:

                # Plots rolling estimates of residuals for each asset-factor combination
                model.plot_rolling_residuals()
                
                df_rsq = pd.DataFrame(data=model.rsquared_adj(), index=[asset], columns=[model.factor_combinations])
                df_asset_rsq = pd.concat([df_asset_rsq, df_rsq])
                
                # Plots of rolling estimates of correlationss between residuals and factors
                model.plot_rolling_correlations_residuals_vs_factor(factor_set=factors)

                # Histogram of rolling estimates of factors grouped into 3 different time periods
                model.plot_rolling_beta_groups()

                # Plots of rolling estimates of factors
                model.plot_rolling_betas()

            else:
                # Partial Regression Plots
                model.partial_regression_plot()
                
                # Factor Loadings
                df_params = pd.DataFrame(model.params).transpose()
                df_params.rename(columns={"const": "Alpha"}, inplace=True)        
                df_params.index = [asset]
                df_asset_params = pd.concat([df_asset_params, df_params])




        ff_asset_rsq = pd.concat([ff_asset_rsq, df_asset_rsq], axis=1)
        ff_asset_params = pd.concat([ff_asset_params, df_asset_params], axis=1)

    return ff_asset_rsq if rolling else ff_asset_params

if __name__ == "__main__":


    # Perform Fama-French Regression Analysis (Single & 6 Factor FF Model)
    famafrench_regression_analysis(asset_type="stock")