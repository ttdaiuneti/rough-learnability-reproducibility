"""IID sampling with replacement from each finite empirical population (Section 7.2).

Nothing here may change the recorded population truths: they are read from
data/processed/<slug>.truth.json, which run_population_truth.py wrote and hashed.

Sampling. A repetition is one multinomial draw of n observations over the support atoms of
P_N; the empirical structures depend on the sample only through which atoms were observed, so
the observed-atom indicator is the sufficient summary. Recovery is then decided by the
edge-realisation criterion derived in exact_rough.py and verified against literal recomputation
in validation.py:

    core recovered   <=>  every singleton edge of the population clutter is realised
    family recovered <=>  every edge of the population clutter is realised

Two stages: the whole grid at 10 000 repetitions, then the threshold-critical grid points
re-run at 100 000 with an independent seed stream. Both stages are kept in the CSV; nothing is
overwritten.

Usage: python3 run_recovery.py [--reps 10000] [--reps-refine 100000]
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

import exact_rough as er
import validation as val

HERE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(HERE, "data", "processed")
RESULTS = os.path.join(HERE, "results")

BASE_ENTROPY = 20260914
FRACTIONS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
LOG_LADDER = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
Q = 0.95
WALL_CLOCK_PER_DATASET = 1800


def n_grid(N):
    ns = {max(1, int(round(f * N))) for f in FRACTIONS}
    ns |= {n for n in LOG_LADDER if n <= 5 * N}
    return sorted(ns)


def load(slug):
    frame = pd.read_csv(os.path.join(PROC, f"{slug}.csv"))
    cols = [c for c in frame.columns if c.startswith("a")]
    pop = er.Population(frame[cols].to_numpy(), frame["y"].to_numpy())
    with open(os.path.join(PROC, f"{slug}.truth.json")) as fh:
        truth = json.load(fh)
    return pop, truth


def simulate(pop, index, core_t, fam_t, n, reps, key):
    rng = np.random.default_rng(np.random.SeedSequence(key))
    t0 = time.perf_counter()
    counts = rng.multinomial(n, pop.masses, size=reps)
    obs = counts > 0
    core_ok = np.ones(reps, dtype=bool)
    for e in core_t:
        core_ok &= er.realised(obs, index[int(e)])
    fam_ok = np.ones(reps, dtype=bool)
    for e in fam_t:
        fam_ok &= er.realised(obs, index[int(e)])
    return int(core_ok.sum()), int(fam_ok.sum()), time.perf_counter() - t0


def row_for(slug, group, N, n, target, hits, reps, key, secs, stage):
    p_hat = hits / reps
    lo, hi = val.wilson(hits, reps)
    return dict(dataset=slug, group=group, N=N, n=n, n_over_N=n / N, target=target,
                stage=stage, reps=reps, successes=hits, p_hat=p_hat,
                se=val.mc_se(p_hat, reps), wilson_lo=lo, wilson_hi=hi,
                seed=str(key), runtime_sec=round(secs, 4))


def run(reps_default, reps_refine):
    truth_tbl = pd.read_csv(os.path.join(RESULTS, "population_truth.csv"))
    rows = []
    for di, rec in truth_tbl.iterrows():
        slug, group, N = rec.dataset, rec.group, int(rec.N)
        pop, truth = load(slug)
        core_t = np.array(truth["core_target_edges"], dtype=np.int64)
        fam_t = np.array(truth["family_target_edges"], dtype=np.int64)
        index = er.edge_pair_index(pop, fam_t)
        ns = n_grid(N)
        t_start = time.perf_counter()
        for gi, n in enumerate(ns):
            key = [BASE_ENTROPY, int(di), gi, reps_default]
            c, f, secs = simulate(pop, index, core_t, fam_t, n, reps_default, key)
            rows.append(row_for(slug, group, N, n, "core", c, reps_default, key, secs,
                                "initial"))
            rows.append(row_for(slug, group, N, n, "family", f, reps_default, key, secs,
                                "initial"))
            if time.perf_counter() - t_start > WALL_CLOCK_PER_DATASET:
                print(f"    wall clock exceeded for {slug} at n={n}; grid truncated",
                      flush=True)
                break
        print(f"  {slug:<26} group {group}  grid {len(ns)} points  "
              f"{time.perf_counter() - t_start:.1f}s", flush=True)

    df = pd.DataFrame(rows)

    # ---- refine the threshold-critical points at the larger repetition budget
    refine = []
    for (slug, target), g in df.groupby(["dataset", "target"]):
        g = g.sort_values("n")
        below = g[g.p_hat < Q]
        above = g[g.p_hat >= Q]
        crit = set()
        if len(above):
            crit.add(int(above.n.iloc[0]))
        if len(below):
            crit.add(int(below.n.iloc[-1]))
        for n in sorted(crit):
            rec = truth_tbl[truth_tbl.dataset == slug].iloc[0]
            di = int(truth_tbl.index[truth_tbl.dataset == slug][0])
            pop, truth = load(slug)
            core_t = np.array(truth["core_target_edges"], dtype=np.int64)
            fam_t = np.array(truth["family_target_edges"], dtype=np.int64)
            index = er.edge_pair_index(pop, fam_t)
            gi = n_grid(int(rec.N)).index(n)
            key = [BASE_ENTROPY, di, gi, reps_refine, 7]
            c, f, secs = simulate(pop, index, core_t, fam_t, n, reps_refine, key)
            hits = c if target == "core" else f
            refine.append(row_for(slug, rec.group, int(rec.N), n, target, hits, reps_refine,
                                  key, secs, "refine"))
    if refine:
        df = pd.concat([df, pd.DataFrame(refine)], ignore_index=True)
        print(f"  refined {len(refine)} threshold-critical points at {reps_refine} reps")

    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(os.path.join(RESULTS, "recovery_results.csv"), index=False)
    return df


# --------------------------------------------------------------------------- thresholds
def best_rows(df):
    """One row per (dataset, target, n): the one with the most repetitions."""
    return df.sort_values("reps").drop_duplicates(subset=["dataset", "target", "n"],
                                                  keep="last")


def thresholds(df, truth_tbl):
    best = best_rows(df)
    out = []
    for (slug, target), g in best.groupby(["dataset", "target"]):
        g = g.sort_values("n")
        n_max = int(g.n.max())
        pt = g[g.p_hat >= Q]
        cert = g[g.wilson_lo >= Q]
        rec = truth_tbl[truth_tbl.dataset == slug].iloc[0]
        out.append(dict(
            dataset=slug, group=rec.group, target=target, n_max=n_max,
            n_point=int(pt.n.iloc[0]) if len(pt) else np.nan,
            n_certified=int(cert.n.iloc[0]) if len(cert) else np.nan,
            censored_point=not len(pt), censored_certified=not len(cert),
            reps=int(g.reps.max())))
    t = pd.DataFrame(out)
    t.to_csv(os.path.join(RESULTS, "threshold_results.csv"), index=False)
    return t


def summary_table(thr, truth_tbl):
    """The per-dataset summary reported as Table 4 of the paper."""
    rows = []
    for slug, g in thr.groupby("dataset"):
        core = g[g.target == "core"].iloc[0]
        fam = g[g.target == "family"].iloc[0]
        rec = truth_tbl[truth_tbl.dataset == slug].iloc[0]
        same_event = int(rec.n_core_target_edges) == int(rec.n_family_target_edges)
        s_point = (fam.n_point / core.n_point
                   if not (core.censored_point or fam.censored_point) else np.nan)
        s_cert = (fam.n_certified / core.n_certified
                  if not (core.censored_certified or fam.censored_certified) else np.nan)
        rows.append(dict(
            dataset=slug, group=rec.group, n_reducts=int(rec.n_reducts),
            core_size=int(rec.core_size),
            identical_recovery_event=bool(same_event),
            n_core_point=core.n_point, n_family_point=fam.n_point,
            n_core_certified=core.n_certified, n_family_certified=fam.n_certified,
            censored_core=bool(core.censored_certified), censored_family=bool(fam.censored_certified),
            S_point=s_point, S_certified=s_cert, reps=int(g.reps.max()), n_max=int(core.n_max)))
    t = pd.DataFrame(rows).sort_values(["group", "dataset"])
    os.makedirs(RESULTS, exist_ok=True)
    t.to_csv(os.path.join(RESULTS, "recovery_summary.csv"), index=False)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--reps-refine", type=int, default=100000)
    args = ap.parse_args()
    truth_tbl = pd.read_csv(os.path.join(RESULTS, "population_truth.csv"))
    print("IID sampling with replacement from each finite empirical population")
    df = run(args.reps, args.reps_refine)
    thr = thresholds(df, truth_tbl)
    summ = summary_table(thr, truth_tbl)
    print()
    print(summ[["dataset", "group", "n_reducts", "core_size", "identical_recovery_event",
                "n_core_certified", "n_family_certified", "censored_family",
                "S_certified"]].to_string(index=False))


if __name__ == "__main__":
    main()
