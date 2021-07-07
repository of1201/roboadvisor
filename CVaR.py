"""
robustMVO.py>
"""
import numpy as np
import math
from scipy.optimize import minimize
from scipy.stats.distributions import chi2
from numpy import linalg
from scipy.stats import gmean
from scipy.optimize import linprog

# ----- function: CVaR model ------
"""
min     gamma + (1 / [(1 - alpha) * S]) * sum( z_s )
s.t.    z_s   >= 0,                 for s = 1, ..., S
        z_s   >= -r_s' x - gamma,   for s = 1, ..., S
        1' x  =  1,
        mu' x >= R
"""

def CVaR(returns, alpha):
    mu = gmean(returns+1) - 1  # Estimate the geometric mean
    R = 1.1 * np.mean(mu)  # Set our target return
    t, p = returns.shape  # Determine the number of assets and scenarios
    # formulate the linear program
    # bound
    lst_t = np.ones(t)
    lst_p = np.ones(p)
    bounds = tuple((0, None) for i in lst_p) + tuple((0, None) for i in lst_t) + \
             tuple((-math.inf, None) for i in np.ones(1))
    # inequality constraint matrix A and vector b
    A = np.vstack((np.hstack((-returns, -np.identity(t), -np.ones((t, 1)))),
                  np.hstack((-mu[np.newaxis], np.zeros((1, t)), np.zeros((1, 1))))))
    b = np.vstack((np.zeros((t, 1)), -R[np.newaxis]))
    # equality constraint matrix Aeq and vector beq
    Aeq = np.hstack((np.ones((1, p)), np.zeros((1, t)), np.zeros((1, 1))))
    beq = 1
    # coefficient of variables
    k = 1 / ((1 - alpha) * t)
    c = np.vstack((np.zeros((p, 1)), k*np.ones((t, 1)), np.ones((1, 1))))
    # linprog optimizer
    res = linprog(c, A_ub=A, b_ub=b, A_eq=Aeq, b_eq=beq, bounds=bounds)
    x = np.asmatrix(res.x[:p])

    return x