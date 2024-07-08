import numpy as np
from numpy.linalg import multi_dot
import scipy.optimize as opt



class PortoflioOptimizer:
    """
    Optimize the Mean-Variance objective function subject to the constraint given by the Budget Equation:

    w' * cov * w - mu' * w; 

    s.t. w' * I = 1

    where   w - Weights of Portfolio
            cov - Covariance Matrix obtained from Annual Historical Returns of Portfolio
            mu - Annualized Historical Returns of Portfolio
            w', mu' are the transposed vectors of w, mu respectively
    
    Optimize the Max Sharpe Ratio objective function subject to the constraint given by the Budget Equation:

    (w' * mu - rf) / sqrt(w' * cov * w)

    s.t. w' * I = 1

    where   w - Weights of Portfolio
            cov - Covariance Matrix obtained from Annual Historical Returns of Portfolio
            mu - Annualized Historical Returns of Portfolio
            rf - Annualized Risk Free Returns
            w' is the transposed vector of w

    """
    def __init__(self, mu, cov, tr, rf, risk_aversion):
        self.mu = mu
        self.cov = cov
        self.risk_aversion = risk_aversion
        self.I = np.ones(cov.shape[0])
        self.tr = tr
        self.rf = rf

    @property
    def cov_inv(self):
        try:
            return np.linalg.inv(self.cov)
        except np.linalg.LinAlgError:
            print("Inverse of Covariance Matrix does not exist: Singular Matrix!")


    def risk_adjusted_returns(self, w):
        """
        Objective Function : Risk Adjusted Returns
        
        0.5 * w' * cov * w - mu' * w; 

        """

        return 0.5 * self.risk_aversion * multi_dot([w.T, self.cov, w]) - np.dot(self.mu.T, w)
    
    def volatility(self, w):
        """
        Objective Function : Risk Adjusted Returns
        
        w' * cov * w;

        """

        return np.sqrt(multi_dot([w.T, self.cov, w]))

    def sharpe_ratio(self, w):
        """
        Objective Function : Sharpe Ratio 
        
        (w' * mu - r) / sqrt(w' * cov * w)
        """

        return -(np.dot(w.T, self.mu) - self.rf) / np.sqrt(multi_dot([w.T, self.cov, w])) 


    def budget_constraint(self, w):
        """
        Budget Constaint equation

        w' * I = 1
        """

        return np.dot(w.T, self.I) - 1
    
    def target_return_constraint(self, w):
        """
        Target Return constraint

        w'mu = r
        """

        return np.dot(w.T, self.mu) - self.tr
    
    def bounds(self, w):
        """
        Bounds on weights -- avoid short selling
        """
        return tuple((0, 1) for _ in range(len(w)))

    def optimize(self, method):
        """
        Optimization of weights of portfolio using scoipy.optimize.minimize
        """

        # constraints
        cons = (
            {'type': 'eq', 'fun': self.budget_constraint},
            {'type': 'eq', 'fun': self.target_return_constraint},            
            )

        # Initial guess of weights (equal probability)
        w0 = np.ones(self.mu.shape[0])*(1/self.mu.shape[0])

        objective_function = {"volatility": self.volatility, "risk adjusted returns": self.risk_adjusted_returns, "sharpe ratio": self.sharpe_ratio}

        return opt.minimize(objective_function[method], w0, method="SLSQP", constraints=cons)




class MeanVarOptimizer:
    """
    Optimize the Mean-Variance objective function subject to the constraint given by the Budget Equation:

    w' * cov * w - mu' * w; 

    s.t. w' * I = 1

    where   w - Weights of Portfolio
            cov - Covariance Matrix obtained from Annual Historical Returns of Portfolio
            mu - Annualized Historical Returns of Portfolio
            w', mu' are the transposed vectors of w, mu respectively
    """

    def __init__(self, mu, cov, r, risk_aversion):
        self.mu = mu
        self.cov = cov
        self.risk_aversion = risk_aversion
        self.I = np.ones(cov.shape[0])
        self.r = r

    @property
    def cov_inv(self):
        try:
            return np.linalg.inv(self.cov)
        except np.linalg.LinAlgError:
            print("Inverse of Covariance Matrix does not exist: Singular Matrix!")


    def objective_function(self, w):
        """
        Objective Function
        
        0.5 * w' * cov * w - mu' * w; 

        """

        return 0.5 * self.risk_aversion * multi_dot([w.T, self.cov, w]) - np.dot(self.mu.T, w)
    

    def budget_constraint(self, w):
        """
        Budget Constaint equation

        w' * I = 1
        """

        return np.dot(w.T, self.I) - 1
    
    def target_return_constraint(self, w):
        """
        Target Return constraint

        w'mu = r
        """

        return np.dot(w.T, self.mu) - self.r
    
    @property
    def optimal_w(self):
        """
        Optimization of weights of portfolio using scoipy.optimize.minimize
        """
        # constraints
        cons = (
            {'type': 'eq', 'fun': self.budget_constraint},
            {'type': 'eq', 'fun': self.target_return_constraint},            
            )

        # Initial guess of weights (equal probability)
        w0 = np.ones(self.mu.shape[0])*(1/self.mu.shape[0])

        return opt.minimize(self.objective_function, w0, constraints=cons)['x']
    

    def equilibrium_returns(self):
        """
        Calculate Equilibrium Risk Premia
        """
        return self.risk_aversion * np.dot(self.cov_inv, self.optimal_w)
        



class MaxSharpeRatioOptimizer:
    """
    Optimize the Max Sharpe Ratio objective function subject to the constraint given by the Budget Equation:

    (w' * mu - rf) / sqrt(w' * cov * w)

    s.t. w' * I = 1

    where   w - Weights of Portfolio
            cov - Covariance Matrix obtained from Annual Historical Returns of Portfolio
            mu - Annualized Historical Returns of Portfolio
            rf - Annualized Risk Free Returns
            w' is the transposed vector of w
    """
    def __init__(self, mu, cov, rf):
        self.mu = mu
        self.cov = cov
        self.rf = rf
        self.I = np.ones(cov.shape[0])


    def objective_function(self, w):
        """
        Objective Function
        
        (w' * mu - r) / sqrt(w' * cov * w)
        """
        return -(np.dot(w.T, self.mu) - self.rf) / np.sqrt(multi_dot([w.T, self.cov, w])) 
    

    def budget_constraint(self, w):
        """
        Budget Constaint equation

        w' * I = 1
        """

        return np.dot(w.T, self.I) - 1
    
    def target_return_constraint(self, w):
        """
        Target Return constraint

        w'mu = r
        """

        return np.dot(w.T, self.mu) - self.r


    @property
    def optimal_w(self):
        """
        Optimization of weights of portfolio using scoipy.optimize.minimize
        """
        # constraints
        cons = (
            {'type': 'eq', 'fun': self.budget_constraint},
            {'type': 'eq', 'fun': self.target_return_constraint},            
            )

        # Initial guess of weights (equal probability)
        w0 = np.ones(self.mu.shape[0])*(1/self.mu.shape[0])

        return opt.minimize(self.objective_function, w0, constraints=cons)['x']

    
