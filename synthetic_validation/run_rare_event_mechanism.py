"""The rare-event mechanism behind the core-family separation (Section 4.1, 5.3 of the paper).

Split in two:

PART 1 -- RARE-TIME VALIDATION. Does the first rare-atom time T_rare = inf{n : X_n = x3}
satisfy P(T_rare > n) = (1-eps)^n?

    Two samplers are run and both are reported:

      categorical  draw from the 4-way law of the construction by inverse CDF and read T_rare
                   off the realised sequence. THIS is the evidential path: the survival law
                   comes out as a measured consequence of the data-generating process.
      geometric    draw T_rare ~ Geometric(eps) directly. Cheap and RAM-free, but it cannot
                   by itself confirm the survival law -- (1-eps)^n IS the survival function of
                   the law being drawn from, so that check is circular and would only exercise
                   numpy's sampler. It is kept as a sampler-equivalence cross-check, and is
                   labelled as such in the CSV.

    The categorical sampler is not expensive: draws are generated in adaptive blocks (at most
    2e6 elements at a time, int8) and only for repetitions that have not yet seen x3, so total
    work is about reps/eps draws and peak memory is a few MB, not reps x O(1/eps).

PART 2 -- STRUCTURAL TRANSITION VALIDATION. On a modest number of fully sequential
trajectories under P0, record the first-hit time of every atom and therefore

    t_core   = the first n at which the empirical core is exactly CORE_P,
    t_family = the first n at which the empirical family is exactly R_P,
    T_rare   = the first n at which x3 appears,

then verify by recomputation (not by assumption) that the empirical family is wrong at
t_family - 1 and right at t_family. This is the combinatorial certificate behind the theorem:
the core witness is {x0, x2} and the family witness is {x0, x2, x3}, so t_family = max(t_core,
T_rare) and the whole eps-dependence enters through T_rare alone.

It also blocks an easy overstatement: seeing x3 is NECESSARY for exact family recovery under
P0 but NOT SUFFICIENT -- when x0 or x2 is still unseen at that moment the family is still
wrong. The run reports how often that happens.

Usage: python3 run_rare_event_mechanism.py [--reps 10000] [--traj 2000]
"""
import argparse
import math
import os
import time

import numpy as np
import pandas as pd

import analytic_formulas as af
import rough_exact as rx
import validation as val

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

BASE_ENTROPY = 20260913
EPS_GRID = [0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
SURV_N = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]


# --------------------------------------------------------------------------- samplers
def first_hit_times(eps, reps, rng, cap, atoms=(3,)):
    """First-hit time (1-indexed) of each atom in `atoms`, per repetition.

    Adaptive blocks: at most `budget` draws are materialised at once, and only for repetitions
    that have not yet completed, so peak memory does not grow as eps shrinks. Repetitions that
    never complete within `cap` carry cap+1 and are counted as censored."""
    m = rx.masses(eps)
    cum = np.cumsum(m)
    first = np.full((reps, len(rx.SUPPORT)), cap + 1, dtype=np.int64)
    active = np.arange(reps)
    offset, budget = 0, 2_000_000
    while active.size and offset < cap:
        b = int(max(1, min(cap - offset, budget // active.size)))
        draws = np.searchsorted(cum, rng.random((active.size, b))).astype(np.int8)
        for a in range(len(rx.SUPPORT)):
            hit = draws == a
            has = hit.any(axis=1)
            t = offset + np.where(has, hit.argmax(axis=1), b) + 1
            pending = first[active, a] > cap
            upd = has & pending
            first[active[upd], a] = t[upd]
        done = (first[active][:, list(atoms)] <= cap).all(axis=1)
        active = active[~done]
        offset += b
    return first


def survival_rows(sampler, eps, t_rare, reps, censored, seed, secs):
    rows = []
    for n in SURV_N:
        k = int((t_rare > n).sum())
        p_hat = k / reps
        ref = af.rare_survival(n, eps)
        lo, hi = val.wilson(k, reps)
        tol, err, z, ok = val.agreement(p_hat, ref, reps)
        rows.append(dict(part="1 rare-time", sampler=sampler,
                         metric="survival_P(T_rare>n)", eps=eps, n=n, reps=reps,
                         successes=k, p_hat=p_hat, se=val.mc_se(p_hat, reps),
                         wilson_lo=lo, wilson_hi=hi, p_analytic=ref, abs_err=err,
                         tolerance=tol, z_score=z, within_tolerance=bool(ok),
                         censored=censored, seed=seed, runtime_sec=round(secs, 3)))
    return rows


# --------------------------------------------------------------------------- run
def run(reps, traj):
    rows = []
    for i_eps, eps in enumerate(EPS_GRID):
        cap = int(math.ceil(25.0 / eps))

        # ---- part 1a: categorical sampler (evidential)
        rng = np.random.default_rng(np.random.SeedSequence([BASE_ENTROPY, i_eps, reps, 1]))
        t0 = time.perf_counter()
        first = first_hit_times(eps, reps, rng, cap, atoms=(3,))
        secs = time.perf_counter() - t0
        t_rare = first[:, 3]
        cens = int((t_rare > cap).sum())
        rows += survival_rows("categorical", eps, t_rare, reps, cens,
                              str([BASE_ENTROPY, i_eps, reps, 1]), secs)

        # ---- part 1b: direct geometric sampler (cross-check only, cannot confirm the law)
        rng_g = np.random.default_rng(np.random.SeedSequence([BASE_ENTROPY, i_eps, reps, 2]))
        t0 = time.perf_counter()
        t_geom = rng_g.geometric(eps, size=reps)
        rows += survival_rows("geometric", eps, t_geom, reps, 0,
                              str([BASE_ENTROPY, i_eps, reps, 2]), time.perf_counter() - t0)

        # ---- part 2: structural transition on representative sequential trajectories
        rng_t = np.random.default_rng(np.random.SeedSequence([BASE_ENTROPY, i_eps, traj, 3]))
        t0 = time.perf_counter()
        fh = first_hit_times(eps, traj, rng_t, cap, atoms=(0, 1, 2, 3))
        secs_t = time.perf_counter() - t0
        seed_t = str([BASE_ENTROPY, i_eps, traj, 3])
        t_core = fh[:, [0, 2]].max(axis=1)
        t_fam = fh[:, [0, 2, 3]].max(axis=1)
        ok_traj = (t_fam <= cap)
        nok = int(ok_traj.sum())

        # recompute the structures at t_family and t_family - 1 from the atoms seen by then
        def seen_mask(t):
            return (fh <= t[:, None])
        at = seen_mask(t_fam)[ok_traj]
        before = seen_mask(t_fam - 1)[ok_traj]
        fam_at = np.array([rx.recovery(0, frozenset(np.flatnonzero(r)))[1] for r in at])
        fam_before = np.array([rx.recovery(0, frozenset(np.flatnonzero(r)))[1] for r in before])
        core_at_trare = np.array([rx.recovery(0, frozenset(np.flatnonzero(r)))[1]
                                  for r in seen_mask(fh[:, 3])[ok_traj]])

        for metric, k, note in (
                ("family_correct_at_t_family", int(fam_at.sum()),
                 "must be 1.0: recovery time is exactly the witness-completion time"),
                ("family_correct_at_t_family_minus_1", int(fam_before.sum()),
                 "must be 0.0: the family is still wrong one draw earlier"),
                ("family_correct_at_first_rare_atom", int(core_at_trare.sum()),
                 "NOT 1.0: seeing x3 is necessary but not sufficient"),
                ("t_family_strictly_after_T_rare", int((t_fam > fh[:, 3])[ok_traj].sum()),
                 "trajectories where x0 or x2 was still unseen when x3 arrived")):
            lo, hi = val.wilson(k, nok)
            rows.append(dict(part="2 structural transition", sampler="categorical",
                             metric=metric, eps=eps, n=np.nan, reps=nok, successes=k,
                             p_hat=k / nok, se=val.mc_se(k / nok, nok), wilson_lo=lo,
                             wilson_hi=hi, p_analytic=np.nan, abs_err=np.nan,
                             tolerance=np.nan, z_score=np.nan, within_tolerance=True,
                             censored=int((~ok_traj).sum()), seed=seed_t,
                             runtime_sec=round(secs_t, 3), note=note))
        for metric, arr in (("median_t_core", t_core[ok_traj]),
                            ("median_t_family", t_fam[ok_traj]),
                            ("median_T_rare", fh[ok_traj, 3])):
            rows.append(dict(part="2 structural transition", sampler="categorical",
                             metric=metric, eps=eps, n=np.nan, reps=nok,
                             successes=np.nan, p_hat=float(np.median(arr)),
                             se=np.nan, wilson_lo=float(np.quantile(arr, 0.25)),
                             wilson_hi=float(np.quantile(arr, 0.75)), p_analytic=np.nan,
                             abs_err=np.nan, tolerance=np.nan, z_score=np.nan,
                             within_tolerance=True, censored=int((~ok_traj).sum()),
                             seed=seed_t, runtime_sec=round(secs_t, 3),
                             note="median with inter-quartile range in the wilson_* columns"))
        # all three medians below come from the SAME part-2 trajectories, so they are
        # comparable (t_family >= T_rare holds pointwise); part 1 has its own larger sample
        print(f"  eps = {eps:<7} [part 2, n = {nok}] median T_rare = "
              f"{int(np.median(fh[ok_traj, 3])):<6} median t_core = "
              f"{int(np.median(t_core[ok_traj])):<4} median t_family = "
              f"{int(np.median(t_fam[ok_traj])):<6} censored = {cens}", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(os.path.join(RESULTS, "rare_event_mechanism.csv"), index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--traj", type=int, default=2000)
    args = ap.parse_args()
    print("rare-event mechanism")
    df = run(args.reps, args.traj)
    for sampler in ("categorical", "geometric"):
        s = df[(df.metric == "survival_P(T_rare>n)") & (df.sampler == sampler)]
        bad = s[~s.within_tolerance]
        print(f"{sampler:<12} {len(s)} survival points, {len(bad)} outside tolerance")
        if len(bad):
            print(bad[["eps", "n", "p_hat", "p_analytic", "abs_err", "tolerance",
                       "z_score"]].to_string(index=False))
    p2 = df[df.part == "2 structural transition"]
    for m in ("family_correct_at_t_family", "family_correct_at_t_family_minus_1",
              "family_correct_at_first_rare_atom"):
        s = p2[p2.metric == m]
        print(f"{m:<36} min {s.p_hat.min():.4f}  max {s.p_hat.max():.4f}")


if __name__ == "__main__":
    main()
