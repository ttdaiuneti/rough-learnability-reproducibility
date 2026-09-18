"""Finite-sample recovery of the population core and of the complete reduct family.

Monte Carlo over the three-population class of Section 4.1, compared against the exact
analytic curves of Propositions 3-5. Raw aggregate results are written to
results/core_family_recovery.csv.

Sampling. Labels are deterministic, so a repetition affects the empirical structures only
through the SET of atoms it observed. One repetition is therefore one multinomial count vector
over (x0, x1, x2, x3); the observed set is (counts > 0), and the empirical family/core come
from the enumerated table in rough_exact.py. validation.py checks this lookup against the
literal labeled-sample computation on random samples.

Seeds. Every point draws from numpy.random.default_rng(SeedSequence([BASE, theta, i_eps, i_n,
reps])), so the whole grid is reproducible point-by-point and the entropy is recorded in the
CSV. Re-running one point does not require re-running the grid.

Usage: python3 run_core_family_recovery.py [--reps 10000] [--reps-hard 100000]
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

import analytic_formulas as af
import rough_exact as rx
import validation as val

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

BASE_ENTROPY = 20260912
EPS_GRID = [0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
BASE_N = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
EXTRA_N = [10000, 20000, 50000]


def n_grid_for(eps):
    """Base grid, extended while the exact family curve has not yet reached 0.99."""
    ns = list(BASE_N)
    for n in EXTRA_N:
        if af.plugin_recovery(0, "family", ns[-1], eps) < 0.99:
            ns.append(n)
    return ns


def hard_points(eps, ns):
    """Points that get the larger repetition budget: the steepest part of each curve, where
    Monte Carlo noise is largest, is where an implementation error would actually show."""
    hard = set()
    for theta in rx.THETAS:
        for target, ref in (("core", lambda n: af.spec_core_recovery(theta, n, eps)),
                            ("family", lambda n: af.plugin_recovery(theta, "family", n, eps))):
            hard.add((theta, target, min(ns, key=lambda n: abs(ref(n) - 0.5))))
    return hard


def simulate(theta, eps, n, reps, seed_key):
    """Return (core successes, family successes, runtime seconds)."""
    lut_core = np.array([rx.recovery(theta, T)[0] for T in _ORDERED_SETS])
    lut_fam = np.array([rx.recovery(theta, T)[1] for T in _ORDERED_SETS])
    rng = np.random.default_rng(np.random.SeedSequence(seed_key))
    t0 = time.perf_counter()
    counts = rng.multinomial(n, rx.masses(eps), size=reps)
    idx = (counts > 0) @ np.array([1, 2, 4, 8])
    core_hits = int(lut_core[idx].sum())
    fam_hits = int(lut_fam[idx].sum())
    return core_hits, fam_hits, time.perf_counter() - t0


# index i of _ORDERED_SETS is the bitmask of observed atoms, matching (counts>0) @ [1,2,4,8]
_ORDERED_SETS = [frozenset(j for j in range(4) if i >> j & 1) for i in range(16)]


def make_row(theta, eps, n, target, hits, reps, hard, key, secs, stage):
    p_hat = hits / reps
    ref = af.spec_core_recovery(theta, n, eps) if target == "core" \
        else af.plugin_recovery(theta, "family", n, eps)
    lo, hi = val.wilson(hits, reps)
    tol, err, z, ok = val.agreement(p_hat, ref, reps)
    return dict(
        eps=eps, n=n, theta=f"P{theta}", target=target, stage=stage,
        reps=reps, successes=hits, p_hat=p_hat, se=val.mc_se(p_hat, reps),
        wilson_lo=lo, wilson_hi=hi, p_analytic=ref,
        analytic_tier="closed form (Prop. 3-5)" if target == "core" else "plug-in estimator",
        abs_err=err, tolerance=tol, z_score=z, within_tolerance=bool(ok),
        hard_point=hard, seed=str(key), runtime_sec=round(secs, 4))


def save(df):
    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(os.path.join(RESULTS, "core_family_recovery.csv"), index=False)
    return df


def run(reps_default, reps_hard):
    rows = []
    for i_eps, eps in enumerate(EPS_GRID):
        ns = n_grid_for(eps)
        hard = hard_points(eps, ns)
        for i_n, n in enumerate(ns):
            for theta in rx.THETAS:
                reps = reps_hard if any(h[0] == theta and h[2] == n for h in hard) \
                    else reps_default
                key = [BASE_ENTROPY, theta, i_eps, i_n, reps]
                c_hits, f_hits, secs = simulate(theta, eps, n, reps, key)
                for target, hits in (("core", c_hits), ("family", f_hits)):
                    rows.append(make_row(theta, eps, n, target, hits, reps,
                                         (theta, target, n) in hard, key, secs, "initial"))
        print(f"  eps = {eps:<7} n up to {ns[-1]:<6} done", flush=True)
    return save(pd.DataFrame(rows))


def rerun_failures(df, factor=10):
    """Re-run every out-of-tolerance point with `factor` times the repetitions and an
    independent seed stream.

    The protocol requires this before any failure is declared: at 10^4 repetitions a point in
    the steep part of a curve sits about one standard error from its tolerance, so a single
    exceedance is far more likely to be Monte Carlo noise than a defect. Nothing is deleted --
    the original row stays in the CSV with stage = 'initial', and the verdict is read off the
    'rerun' row."""
    bad = df[(df.stage == "initial") & (~df.within_tolerance)]
    if bad.empty:
        return df
    rows = []
    for _, r in bad.iterrows():
        theta, eps, n = int(r.theta[1:]), float(r.eps), int(r.n)
        reps = int(r.reps) * factor
        i_eps, i_n = EPS_GRID.index(eps), n_grid_for(eps).index(n)
        key = [BASE_ENTROPY, theta, i_eps, i_n, reps, 7]  # 7 = independent rerun stream
        c_hits, f_hits, secs = simulate(theta, eps, n, reps, key)
        hits = c_hits if r.target == "core" else f_hits
        rows.append(make_row(theta, eps, n, r.target, hits, reps, bool(r.hard_point),
                             key, secs, "rerun"))
        print(f"  rerun eps={eps} n={n} {r.theta} {r.target} at {reps} reps: "
              f"p_hat {r.p_hat:.5f} -> {hits / reps:.5f} (analytic {r.p_analytic:.5f})",
              flush=True)
    return save(pd.concat([df, pd.DataFrame(rows)], ignore_index=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--reps-hard", type=int, default=100000)
    args = ap.parse_args()
    print("Monte Carlo grid")
    df = run(args.reps, args.reps_hard)
    init = df[df.stage == "initial"]
    bad = init[~init.within_tolerance]
    print(f"{len(init)} points, {len(bad)} outside tolerance on the first pass")
    if len(bad):
        print(bad[["eps", "n", "theta", "target", "p_hat", "p_analytic", "abs_err",
                   "tolerance", "z_score"]].to_string(index=False))
        df = rerun_failures(df)
        re = df[df.stage == "rerun"]
        print(f"{len(re[~re.within_tolerance])} of {len(re)} rerun points still outside "
              f"tolerance")
    print(f"{len(df)} rows -> results/core_family_recovery.csv")


if __name__ == "__main__":
    main()
