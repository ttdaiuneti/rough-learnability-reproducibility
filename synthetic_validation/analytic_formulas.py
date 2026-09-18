"""Analytic probabilities for the three-population construction of Section 4.1.

Two tiers, kept apart on purpose:

TIER 1 -- closed forms stated in the paper. These are the theory under test; the Monte Carlo
          scripts compare simulation against them, and validation.py compares them against
          exact enumeration.
            p_core_p0p1(n)      Proposition 3:   1 - 2(3/4)^n + (1/2)^n
            e_core_p2(n, eps)   Proposition 3:   (3/4)^n + (1/2+eps)^n - (1/4+eps)^n
            core_bound(n)       Corollary 4:     2(3/4)^n
            testing_risk(n,eps) Proposition 7:   (1/2)(1-eps)^n
            rare_survival(n,eps)                 (1-eps)^n
            n_core_ub, n_family_lb               Theorem 9

TIER 2 -- exact recovery probabilities of the PLUG-IN estimator, derived here by enumeration
          over observed-atom sets. These are bookkeeping identities about the constructed
          class, not new theory, and they are labelled as such wherever they appear. In
          particular the plug-in family error is NOT the minimax family risk: Proposition 7
          gives the estimator-independent lower bound (1/2)(1-eps)^n, and the plug-in curve
          sits above it. The two must never be conflated.

All Tier-2 numbers are computed from the structure tables in rough_exact.py, so if the
population definitions ever changed, these formulas would follow rather than silently
disagree.
"""
import math
from fractions import Fraction
from itertools import combinations

import rough_exact as rx


# --------------------------------------------------------------------------- Tier 1
def p_core_p0p1(n):
    """Exact empirical-core recovery for P0 and P1 (uniform in eps)."""
    return 1.0 - 2.0 * (0.75 ** n) + (0.5 ** n)


def e_core_p2(n, eps):
    """Exact empirical-core error for P2."""
    return (0.75 ** n) + ((0.5 + eps) ** n) - ((0.25 + eps) ** n)


def p_core_p2(n, eps):
    return 1.0 - e_core_p2(n, eps)


def core_bound(n):
    """Uniform three-population bound on the core error."""
    return 2.0 * (0.75 ** n)


def testing_risk(n, eps):
    """Exact equal-prior optimal testing risk on the P0/P1 subproblem.

    Lower bound for family minimax recovery over the class; NOT a plug-in curve."""
    return 0.5 * ((1.0 - eps) ** n)


def total_variation(n, eps):
    """TV between the labeled n-fold laws of P0 and P1."""
    return 1.0 - ((1.0 - eps) ** n)


def rare_survival(n, eps):
    """P(the rare atom x3 is unseen in n draws)."""
    return (1.0 - eps) ** n


def n_core_ub(delta):
    """Upper bound on the core sample complexity."""
    return math.ceil(math.log(2.0 / delta) / (-math.log(0.75)))


def n_family_lb(delta, eps):
    """Lower bound on the family sample complexity."""
    return math.ceil(math.log(1.0 / (2.0 * delta)) / (-math.log(1.0 - eps)))


def n_family_small_eps(delta, eps):
    """The small-eps approximation log(1/(2delta))/eps."""
    return math.log(1.0 / (2.0 * delta)) / eps


# --------------------------------------------------------------------------- Tier 2
def _witness_prob(req, n, eps):
    """P(every atom of `req` appears at least once in n IID draws), by inclusion-exclusion on
    the complement -- an alternating sum of at most 2^|req| terms of the form (1 - mass)^n."""
    m = rx.masses(eps)
    acc = 0.0
    for r in range(len(req) + 1):
        for U in combinations(sorted(req), r):
            q = 1.0 - sum(m[i] for i in U)
            acc += ((-1) ** r) * (q ** n)
    return acc


def _general_prob(theta, target, n, eps):
    """Fallback: sum P(observed set = T) over recovering T, with P(observed set = T) itself an
    inclusion-exclusion sum. Numerically worse conditioned than `_witness_prob`; used only as
    a cross-check and when the recovering sets are not a principal up-set."""
    m = rx.masses(eps)
    acc = 0.0
    for T in rx.recovering_sets(theta, target):
        for r in range(len(T) + 1):
            for U in combinations(sorted(T), r):
                acc += ((-1) ** (len(T) - r)) * (sum(m[i] for i in U) ** n)
    return acc


def plugin_recovery(theta, target, n, eps):
    """Exact recovery probability of the plug-in empirical estimator (Tier 2)."""
    req = rx.witness(theta, target)
    if req is None:
        return _general_prob(theta, target, n, eps)
    return _witness_prob(req, n, eps)


def plugin_recovery_exact(theta, target, n, eps_fraction):
    """Same quantity in exact rational arithmetic, for small-n auditing."""
    e = Fraction(eps_fraction)
    m = [Fraction(1, 4), Fraction(1, 2) - e, Fraction(1, 4), e]
    req = rx.witness(theta, target)
    if req is not None:
        acc = Fraction(0)
        for r in range(len(req) + 1):
            for U in combinations(sorted(req), r):
                q = 1 - sum(m[i] for i in U)
                acc += ((-1) ** r) * q ** n
        return acc
    acc = Fraction(0)
    for T in rx.recovering_sets(theta, target):
        for r in range(len(T) + 1):
            for U in combinations(sorted(T), r):
                acc += ((-1) ** (len(T) - r)) * sum((m[i] for i in U), Fraction(0)) ** n
    return acc


def spec_core_recovery(theta, n, eps):
    """The Tier-1 core formula appropriate to the population."""
    return p_core_p0p1(n) if theta in (0, 1) else p_core_p2(n, eps)
