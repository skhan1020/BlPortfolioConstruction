import numpy as np
from numpy.linalg import multi_dot

class BlackLittermanModel:
    def __init__(self, cov, weights, risk_aversion, T, position_matrix, return_vector):
        """
        Implementation of the Black-Litterman Model

        Parameters
        ----------

        cov : pd.DataFrame
            Covariance Matrix of asset returns 
        
        weights : np.array
            Optimized portfolio allocations (prior)
        
        risk_aversion : float
            Investor's risk appetite
        
        T : int
            Number of historical data points
        
        position_matrix : np.ndarray
            Positions taken by investor based on Absolute & Relative Views

        return_vector : np.array
            Expected confidence scores associated with each view of the investor
        
        """

        self.cov = cov
        self.weights = weights
        self.risk_aversion = risk_aversion
        self.tau = 1 / T
        self.P = position_matrix
        self.Q = return_vector

    
    @property
    def excess_return(self):
        """
        
        Equilibrium of Excess Return - Prior View on Risk Premia
        
        """
        
        return self.risk_aversion*np.dot(self.cov, self.weights)

    @property
    def prior_covariance(self):
        """
        
        Prior view on Covariance

        """
        return self.tau * self.cov
    
    @property
    def prior_distbn_excess_ret(self):
        """
        
        Prior distribution of excess return

        """

        return np.random.normal(loc=self.excess_return, scale=self.excess_return, size=1)
    
    @property
    def omega(self):
        """
        
        He & Litterman's definition of covariance matrix : Omega

        """

        return np.diag(np.diag(multi_dot([self.P, self.prior_covariance, self.P.T])))
    
    @property
    def posterior_return(self):
        """

        Mean of the Posterior distribution of excess returns
    
        """
        
        A = np.linalg.inv(np.linalg.inv(self.prior_covariance) + multi_dot([self.P.T, np.linalg.inv(self.omega), self.P]))
        
        B = np.dot(np.linalg.inv(self.prior_covariance), self.excess_return) + multi_dot([self.P.T, np.linalg.inv(self.omega), self.Q])
        
        return np.dot(A, B)



def calc_asset_allocations(risk_aversion, cov, posterior_returns):
    """
    Compute Updated Allocations using Posterior Distribution
    
    Parameters
    ----------

    risk_aversion : float
        Risk aversion of Investor

    cov : pd.DataFrame
        Historical covariance matrix of asset returns
    
    Returns
    -------

    Updated portfolio allocations based on posterior distribution of excess asset returns

    """
    return (1 / risk_aversion) * (np.dot(np.linalg.inv(cov), posterior_returns))

    