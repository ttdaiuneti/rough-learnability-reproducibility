"""Deterministic sample-complexity separation (Theorem 9 and Table 3 of the paper).

Deterministic: no Monte Carlo anywhere in this stage. It evaluates

    n_core^UB    = ceil( log(2/delta) / -log(3/4) )          uniform in eps
    n_family^LB  = ceil( log(1/(2 delta)) / -log(1-eps) )     estimator-independent
    approx       = log(1/(2 delta)) / eps                     small-eps form
    Q            = n_family^LB / n_core^UB

and reports the slope of n_family^LB in eps by fitting it, rather than asserting -1.

The asymmetry is deliberate: n_core^UB is an UPPER bound attained by an explicit estimator
(the plug-in empirical core), while n_family^LB is a LOWER bound over all estimators, inherited
from the two-point testing argument behind the main separation theorem. Q is therefore a
certified separation, not a comparison of two like-for-like curves.

Usage: python3 run_sample_complexity_bounds.py
"""
import math
import os

import numpy as np
import pandas as pd

import analytic_formulas as af

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

DELTAS = [0.10, 0.05, 0.01]
TABLE_EPS = [0.1, 0.01, 0.001, 0.0001]
CORE_EPS_GRID = [0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
CURVE_EPS = sorted(set(list(np.logspace(math.log10(0.25), -5, 40)) + CORE_EPS_GRID + TABLE_EPS))


def build():
    rows = []
    for delta in DELTAS:
        nc = af.n_core_ub(delta)
        for eps in CURVE_EPS:
            nf = af.n_family_lb(delta, eps)
            approx = af.n_family_small_eps(delta, eps)
            rows.append(dict(delta=delta, eps=eps, n_core_ub=nc, n_family_lb=nf,
                             small_eps_approx=approx,
                             rel_approx_error=(approx - nf) / nf,
                             Q=nf / nc, eps_times_Q=eps * nf / nc))
    df = pd.DataFrame(rows)
    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(os.path.join(RESULTS, "sample_complexity_bounds.csv"), index=False)
    return df


def slope(df, delta, eps_max=1e-2):
    s = df[(df.delta == delta) & (df.eps <= eps_max)]
    return float(np.polyfit(np.log10(s.eps), np.log10(s.n_family_lb), 1)[0])


def table_delta001(df):
    """The reduced table at delta = 0.01 reported as Table 3 of the paper."""
    s = df[(df.delta == 0.01) & (df.eps.isin(TABLE_EPS))].sort_values("eps", ascending=False)
    t = s[["eps", "n_core_ub", "n_family_lb", "Q", "small_eps_approx",
           "rel_approx_error"]].copy()
    os.makedirs(RESULTS, exist_ok=True)
    t.to_csv(os.path.join(RESULTS, "sample_complexity_table_delta001.csv"), index=False)
    return t


def main():
    print("deterministic sample-complexity separation")
    df = build()
    t = table_delta001(df)
    print(t.to_string(index=False))
    for delta in DELTAS:
        s = df[(df.delta == delta) & (df.eps <= 1e-4)]
        print(f"  delta = {delta:g}: fitted small-eps slope = {slope(df, delta):.4f}, "
              f"eps*Q -> {s.eps_times_Q.mean():.4f}")


if __name__ == "__main__":
    main()
