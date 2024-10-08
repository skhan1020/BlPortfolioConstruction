import numpy as np
import pandas as pd

def get_dictionary_of_views(returns_dict):
    """
    Generate Absolute and Relative Views associated with every possible equity in the portfolio

    Parameters
    ----------

    returns_dict : dict
        Dictionary of expected returns predicted by ML model from earnings history
    
    Returns
    -------

    absolute_view_dict: dict
        dictionary of predicted returns for every equity
    
    relative_view_dict: dict
        dicionary of relative returns associated with every pair of equity

    """

    absolute_view_dict, relative_view_dict = dict(), dict()

    print("-" * 50 + "Constructing Absolute & Relative Investor Views!" + "-" * 50)

    for i in returns_dict:
        for j in returns_dict:
            if i != j:
                # Absolute Views
                absolute_view_dict.update({str(i) + " returns" : returns_dict[i]})
                
                # Relative Views
                relative_view_dict.update({str(i) + " outperforms " + str(j) : returns_dict[i] - returns_dict[j]})
    
    print("-" * 50 + "Done!" + "-" * 50)

    return absolute_view_dict, relative_view_dict

def compute_matrices(investor_views, returns):
    """
    Computes position matrix and return vector based on investor views and expected asset predictions 

    Parameters
    ----------

    investor_views : List[str]
        List of investor views used to upudate portfolio allocations
        
    returns : pd.Series
        Average empirical or factor model computed returns of assets
    
    Returns
    -------
        P : np.ndarray
            position_matrix
        
        Q : np.array
            return_vector

    """
    # Obtain indices of every asset
    ticker_idx_dict = {k: idx for idx, k in enumerate(returns.to_dict())}
    
    # Compute absolute & relative views
    absolute_view_dict, relative_view_dict = get_dictionary_of_views(returns_dict=returns.to_dict())
    
    # Initialize position matrix and return vector
    P = np.zeros((len(investor_views), len(returns)))

    Q = np.zeros((len(investor_views)))

    for idx, investor_view in enumerate(investor_views):
        if investor_view in absolute_view_dict:
            
            ticker = investor_view.split(" returns")[0]
            ticker_idx = ticker_idx_dict[ticker]

            # Update P & Q matrices based on absolute views            
            P[idx][ticker_idx] = np.sign(absolute_view_dict[investor_view])
            Q[idx] = absolute_view_dict[investor_view]

        elif investor_view in relative_view_dict:
            
            ticker1 = investor_view.split(" outperforms ")[0]
            ticker2 = investor_view.split(" outperforms ")[-1]
            ticker1_idx, ticker2_idx = ticker_idx_dict[ticker1], ticker_idx_dict[ticker2]
            
            # Update P & Q matrices based on relative views
            P[idx][ticker1_idx], P[idx][ticker2_idx] = np.sign(relative_view_dict[investor_view]), -np.sign(relative_view_dict[investor_view])
            Q[idx] = relative_view_dict[investor_view]

    return P, Q
    


def generate_positions(investor_views, returns):
    """
    Function to Generate the Position and Return Matrices for the Likelihood Function in BL model

    Parameters
    ----------

    investor_views : List[str]
        List of Investor Views 
        Two possible investor types : 
            a) Absolute View : Company 1 returns
            b) Relative View : Company 1 outperforms Company 2
    
    returns : pd.Series
        Average of historical returns of each quity within portfolio : Empirical / Factor Model

    
    Returns
    -------

    position_matrix : np.ndarray
        Positions taken by investor based on Absolute & Relative Views

    return_vector : np.array
        Expected returns (absolute or relative) associated with each view of the investor

    """
 
    # Generates Views    
    position_matrix, return_vector = compute_matrices(investor_views=investor_views, returns=returns)

    return position_matrix, return_vector    
        

    
    