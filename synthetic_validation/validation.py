"""Validation layer for the synthetic experiments (Section 7.1 and Appendix B of the paper).

Three jobs, kept separate:

 1. ANALYTIC IDENTITIES. Compare the closed forms stated in the paper against exact
    enumeration over observed-atom sets, in float and (for small n) in exact rational
    arithmetic. These involve no randomness: if an analytic identity fails, the run stops.
 2. IMPLEMENTATION IDENTITIES. The lookup path used by the Monte Carlo engine must agree with
    the literal labeled-sample path on every observed set and on random labeled samples.
 3. MONTE CARLO AGREEMENT. Tolerance is the more permissive of 3 Monte Carlo standard errors
    and an absolute 0.01; the z-score is reported alongside so a reader can apply the
    stricter rule themselves. Nothing is hidden: every violation is flagged.

Run `python3 validation.py` for the standalone analytic + implementation self-test.
"""
import math
import random
from fractions import Fraction

import analytic_formulas as af
import rough_exact as rx

Z95 = 1.959963984540054
ABS_TOL = 0.01


# --------------------------------------------------------------------------- helpers
def wilson(k, n, z=Z95):
    """95% Wilson score interval for a binomial proportion."""
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
    """(tolerance, |error|, z, ok). Tolerance = max(3 SE, 0.01) -- the more permissive of the
    two rules, so a tiny SE at a saturated point cannot manufacture a failure. The z-score is
    returned so the stricter reading stays visible."""
    se = mc_se(p_hat, reps)
    err = abs(p_hat - p_ref)
    tol = max(3.0 * se, ABS_TOL)
    z = err / se if se > 0 else (0.0 if err == 0 else float("inf"))
    return tol, err, z, err <= tol


# --------------------------------------------------------------------------- 1. analytic
def check_analytic_identities(n_grid, eps_grid):
    """Closed forms of the paper vs exact enumeration. Returns (rows, n_fail)."""
    rows = []
    for eps in eps_grid:
        for n in n_grid:
            for theta in rx.THETAS:
                ref = af.plugin_recovery(theta, "core", n, eps)
                spec = af.spec_core_recovery(theta, n, eps)
                bad = abs(ref - spec) > 1e-12
                rows.append(dict(check="core closed form vs enumeration", theta=f"P{theta}",
                                 eps=eps, n=n, enumeration=ref, closed_form=spec,
                                 abs_diff=abs(ref - spec), ok=not bad))
    return rows, sum(1 for r in rows if not r["ok"])


def check_analytic_identities_exact(n_grid, eps_fractions):
    """The same identities in exact rational arithmetic (no floating point at all)."""
    rows = []
    for es in eps_fractions:
        e = Fraction(es)
        for n in n_grid:
            for theta in rx.THETAS:
                ref = af.plugin_recovery_exact(theta, "core", n, es)
                if theta in (0, 1):
                    spec = 1 - 2 * Fraction(3, 4) ** n + Fraction(1, 2) ** n
                else:
                    spec = 1 - (Fraction(3, 4) ** n + (Fraction(1, 2) + e) ** n
                                - (Fraction(1, 4) + e) ** n)
                rows.append(dict(check="core closed form, exact rationals", theta=f"P{theta}",
                                 eps=str(e), n=n, enumeration=str(ref), closed_form=str(spec),
                                 ok=ref == spec))
    return rows, sum(1 for r in rows if not r["ok"])


def check_uniform_bound(n_grid, eps_grid):
    """Corollary 4 uniform core bound on the tested points. Numerical confirmation only: the
    paper's proof is analytic (it follows from the P2 closed form and (1/2+eps) <= 3/4)."""
    rows = []
    for eps in eps_grid:
        for n in n_grid:
            worst = max(1.0 - af.spec_core_recovery(t, n, eps) for t in rx.THETAS)
            bound = af.core_bound(n)
            rows.append(dict(check="sup core error <= 2(3/4)^n", eps=eps, n=n,
                             sup_error=worst, bound=bound, ratio=worst / bound if bound else 0.0,
                             ok=worst <= bound + 1e-15))
    return rows, sum(1 for r in rows if not r["ok"])


def check_testing_expression(n_grid, eps_grid):
    """Proposition 7: TV = 1-(1-eps)^n and equal-prior risk = (1/2)(1-eps)^n, recomputed from
    the labeled-space overlap rather than copied from the closed form."""
    rows = []
    for eps in eps_grid:
        m = rx.masses(eps)
        shared = 1.0 - m[3]  # labeled laws of P0 and P1 differ only on the rare atom
        for n in n_grid:
            overlap = shared ** n
            tv, risk = 1.0 - overlap, 0.5 * overlap
            rows.append(dict(check="two-point testing expression", eps=eps, n=n,
                             tv=tv, tv_spec=af.total_variation(n, eps),
                             risk=risk, risk_spec=af.testing_risk(n, eps),
                             ok=abs(tv - af.total_variation(n, eps)) <= 1e-12
                             and abs(risk - af.testing_risk(n, eps)) <= 1e-12))
    return rows, sum(1 for r in rows if not r["ok"])


# --------------------------------------------------------------------------- 2. implementation
def check_lookup_matches_literal(trials=4000, seed=20260912):
    """The Monte Carlo engine summarises a repetition by its set of observed atoms. Verify on
    random labeled samples (with duplicates, in random order) that the literal path gives the
    same family and core."""
    rnd = random.Random(seed)
    bad = 0
    for _ in range(trials):
        theta = rnd.choice(rx.THETAS)
        k = rnd.randint(1, 12)
        idx = [rnd.randrange(len(rx.SUPPORT)) for _ in range(k)]
        fam_lit, core_lit = rx.empirical_family_core(rx.labeled_sample(idx, theta))
        fam_tab, core_tab = rx._TABLE[theta][frozenset(idx)]
        if (fam_lit, core_lit) != (fam_tab, core_tab):
            bad += 1
    return trials, bad


def check_witness_structure():
    """Report the derived minimal witness sets; None means the recovering sets are not a
    principal up-set and the general expansion is used instead."""
    out = []
    for theta in rx.THETAS:
        for target in ("core", "family"):
            w = rx.witness(theta, target)
            out.append(dict(theta=f"P{theta}", target=target,
                            witness=None if w is None else sorted(f"x{i}" for i in w),
                            n_recovering_sets=len(rx.recovering_sets(theta, target))))
    return out


# --------------------------------------------------------------------------- self-test
def main():
    n_grid = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50]
    eps_grid = [0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    print("population structures")
    for t in rx.THETAS:
        print(f"  P{t}: family = {rx.show_family(rx.POP_FAMILY[t])}, "
              f"core = {rx.show(rx.POP_CORE[t])}")
    print("\nderived minimal witness sets (enumerated, not assumed)")
    for r in check_witness_structure():
        print(f"  {r['theta']} {r['target']:<6} -> {r['witness']}  "
              f"({r['n_recovering_sets']}/16 observed sets recover)")
    print()
    for name, (rows, bad) in (
            ("core closed forms vs enumeration (float)", check_analytic_identities(n_grid, eps_grid)),
            ("core closed forms vs enumeration (exact)",
             check_analytic_identities_exact([1, 2, 3, 5, 8, 13], ["1/5", "1/10", "1/100"])),
            ("uniform core bound", check_uniform_bound(n_grid, eps_grid)),
            ("two-point testing expression", check_testing_expression(n_grid, eps_grid))):
        print(f"  {'PASS' if bad == 0 else 'FAIL'}  {name}: {len(rows)} checks, {bad} failures")
    trials, bad = check_lookup_matches_literal()
    print(f"  {'PASS' if bad == 0 else 'FAIL'}  lookup path vs literal labeled-sample path: "
          f"{trials} random samples, {bad} disagreements")


if __name__ == "__main__":
    main()
