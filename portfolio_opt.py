# src/portfolio_opt.py
import numpy as np
import cvxpy as cp

def optimize_portfolio(expected_returns, cov_matrix, vol_cap=0.02, max_weight=0.3):
    """
    expected_returns: numpy array (n,)
    cov_matrix: numpy array (n,n)
    vol_cap: max allowed portfolio daily volatility (std)
    max_weight: per-asset max weight (0..1)
    Returns: weights vector
    """
    # ensure numpy array
    mu = np.asarray(expected_returns, dtype=float)
    Sigma = np.asarray(cov_matrix, dtype=float)

    # small jitter to ensure positive semidefiniteness / numerical stability
    eps = 1e-8
    Sigma = Sigma + np.eye(Sigma.shape[0]) * eps

    n = len(mu)
    w = cp.Variable(n)

    # objective: maximize expected return
    ret = mu @ w

    # constraint: variance <= vol_cap^2 (DCP-friendly)
    variance = cp.quad_form(w, Sigma)
    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        w <= max_weight,
        variance <= vol_cap ** 2
    ]

    prob = cp.Problem(cp.Maximize(ret), constraints)
    
    prob.solve(solver=cp.SCS, verbose=False)


    if w.value is None:
        # fallback: equal weight
        return np.ones(n) / n
    return np.array(w.value).clip(0, 1)
