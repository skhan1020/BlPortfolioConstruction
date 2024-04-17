import numpy as np
from numpy.linalg import multi_dot

class BlackLittermanModel:
    def __init__(self, cov, weights, risk_aversion, T, position_matrix, return_vector):
        """
        Implementation of the Black-Litterman Model
        """

        self.cov = cov
        self.weights = weights
        self.risk_aversion = risk_aversion
        self.tau = 1 / T
        self.P = position_matrix
        self.Q = return_vector

    
    @property
    def excess_return(self):
    
        # Equilibrium of Excess Return - Prior View on Risk Premia
        return self.risk_aversion*np.dot(self.cov, self.weights)

    @property
    def prior_covariance(self):

        return self.tau * self.cov
    
    @property
    def prior_distbn_excess_ret(self):
        return np.random.normal(loc=self.excess_return, scale=self.excess_return, size=1)
    
    @property
    def omega(self):
        return np.diag(np.diag(multi_dot([self.P, self.prior_covariance, self.P.T])))
    
    @property
    def posterior_return(self):
        
        A = np.linalg.inv(np.linalg.inv(self.prior_covariance) + multi_dot([self.P.T, np.linalg.inv(self.omega), self.P]))
        
        B = np.dot(np.linalg.inv(self.prior_covariance), self.excess_return) + multi_dot([self.P.T, np.linalg.inv(self.omega), self.Q])
        
        return np.dot(A, B)



def calc_asset_allocations(risk_aversion, cov, posterior_returns):
    """
    Compute Updated Allocations using Posterior Distribution
    """
    return (1 / risk_aversion) * (np.dot(np.linalg.inv(cov), posterior_returns))

    