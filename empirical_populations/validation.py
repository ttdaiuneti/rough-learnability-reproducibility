"""Validation layer for the finite empirical population study (Section 7.2 of the paper).

Three jobs, kept separate:

 1. ROUTE EQUIVALENCE. `exact_rough` offers a direct route (sufficiency rebuilt from the
    labeled observations, by definition) and a clutter route (minimal transversals of the
    discernibility clutter). The experiment uses the clutter route; this module checks it
    against the definition on random observed subsets of every accepted population.

 2. RECOVERY CHARACTERISATION. The sampling experiment does not rebuild the empirical family
    for every repetition. It uses the derived criterion

        core recovered   <=>  every singleton edge of the population clutter is realised
        family recovered <=>  every edge of the population clutter is realised

    which is checked here against literally recomputing the empirical family and core and
    comparing them with the recorded population truth. If this check fails anywhere, the fast
    path is wrong and the experiment must not use it.

 3. MONTE CARLO AGREEMENT. Wilson intervals and the tolerance rule: tolerance is the more
    permissive of 3 standard errors and an absolute 0.01, with the z-score reported alongside
    so the stricter reading stays inspectable.

Run `python3 validation.py` for the standalone self-test over every accepted population.
"""
import math
import os

import numpy as np
import pandas as pd

import exact_rough as er

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
PROC = os.path.join(HERE, "data", "processed")

Z95 = 1.959963984540054
ABS_TOL = 0.01


# --------------------------------------------------------------------------- statistics
def wilson(k, n, z=Z95):
    if n == 0:
        return (0.0, 1.0)
    ph = k / n
    d = 1.0 + z * z / n
    centre = ph + z * z / (2 * n)
    half = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n))
    return ((centre - half) / d, (centre + half) / d)


def mc_se(p_hat, n):
    return math.sqrt(max(p_hat * (1 - p_hat), 0.0) / n)


def agreement(p_hat, p_ref, reps):
    se = mc_se(p_hat, reps)
    err = abs(p_hat - p_ref)
    tol = max(3.0 * se, ABS_TOL)
    z = err / se if se > 0 else (0.0 if err == 0 else float("inf"))
    return tol, err, z, err <= tol


# --------------------------------------------------------------------------- loading
def accepted_datasets():
    path = os.path.join(RESULTS, "screening_results.csv")
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    return df[df.status == "ACCEPTED"].dataset.tolist()


def load_population(slug):
    frame = pd.read_csv(os.path.join(PROC, f"{slug}.csv"))
    cols = [c for c in frame.columns if c.startswith("a")]
    return er.Population(frame[cols].to_numpy(), frame["y"].to_numpy())


# --------------------------------------------------------------------------- checks
def random_subsets(n_atoms, trials, rng):
    """Random non-empty observed-atom subsets, with sizes spread from tiny to nearly full."""
    for _ in range(trials):
        k = int(rng.integers(1, n_atoms + 1))
        yield np.sort(rng.choice(n_atoms, size=k, replace=False))


def check_route_equivalence(pop, trials=60, seed=0):
    """Direct (definition) vs clutter (transversal) route on random observed subsets."""
    rng = np.random.default_rng(seed)
    bad = 0
    for sel in random_subsets(len(pop.X), trials, rng):
        X, y = pop.X[sel], pop.y[sel]
        d_fam, d_core = er.family_core_from_rows(X, y, pop.p)
        sub = er.Population(X, y, p=pop.p)
        edges = sub.edges if len(sub.X) > 1 else np.array([], dtype=np.int64)
        c_fam, c_core = er.family_core_from_edges(edges, pop.p)
        if sorted(int(m) for m in d_fam) != sorted(int(m) for m in c_fam) or d_core != c_core:
            bad += 1
    return trials, bad


def check_recovery_characterisation(pop, trials=200, seed=1):
    """Edge-realisation criterion vs literally recomputing the empirical structures."""
    rng = np.random.default_rng(seed)
    truth_fam = set(int(m) for m in er.minimal_transversals(pop.clutter, pop.p))
    truth_core = er.core_mask(er.minimal_transversals(pop.clutter, pop.p))
    core_t, fam_t = er.recovery_targets(pop)
    index = er.edge_pair_index(pop, np.unique(np.concatenate([core_t, fam_t]))
                               if len(core_t) else fam_t)
    bad = 0
    for sel in random_subsets(len(pop.X), trials, rng):
        obs = np.zeros((1, len(pop.X)), dtype=bool)
        obs[0, sel] = True
        fast_core = all(er.realised(obs, index[int(e)])[0] for e in core_t)
        fast_fam = all(er.realised(obs, index[int(e)])[0] for e in fam_t)
        X, y = pop.X[sel], pop.y[sel]
        sub = er.Population(X, y, p=pop.p)
        edges = sub.edges if len(sub.X) > 1 else np.array([], dtype=np.int64)
        fam, core = er.family_core_from_edges(edges, pop.p)
        slow_fam = set(int(m) for m in fam) == truth_fam
        slow_core = core == truth_core
        if fast_core != slow_core or fast_fam != slow_fam:
            bad += 1
    return trials, bad


def main():
    slugs = accepted_datasets()
    if not slugs:
        print("no accepted datasets yet -- run run_screening.py first")
        return
    print(f"{len(slugs)} accepted populations\n")
    tot_r = tot_rb = tot_c = tot_cb = 0
    for slug in slugs:
        pop = load_population(slug)
        t1, b1 = check_route_equivalence(pop)
        t2, b2 = check_recovery_characterisation(pop)
        tot_r += t1
        tot_rb += b1
        tot_c += t2
        tot_cb += b2
        print(f"  {slug:<26} support={len(pop.X):<6} p={pop.p:<3} "
              f"route {t1 - b1}/{t1}   recovery-criterion {t2 - b2}/{t2}")
    print(f"\n  {'PASS' if tot_rb == 0 else 'FAIL'}  direct vs clutter route: "
          f"{tot_r} random subsets, {tot_rb} disagreements")
    print(f"  {'PASS' if tot_cb == 0 else 'FAIL'}  edge-realisation criterion vs recomputation: "
          f"{tot_c} random subsets, {tot_cb} disagreements")


if __name__ == "__main__":
    main()
