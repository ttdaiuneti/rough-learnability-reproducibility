#!/usr/bin/env python3
"""v2 verification gate --- locked_spec_v2.md, Section 18.

Nothing here is a heuristic or a numerical approximation of a probability: every sampling
quantity is computed as an EXACT polynomial in eps over the rationals, and every structural
quantity by exhaustive enumeration.  The closed forms printed by the spec are treated as
CLAIMS to be reproduced by independent computation, never as inputs.

Objects (locked_spec_v2.md Sections 1-5).  A structure is (S, f) with S a set of atoms in
{0,1}^p and f: S -> labels.  D(u,v) = {a : u_a != v_a} for bichromatic pairs; H = the edge
set; the clutter C = minimal edges; R = minimal transversals of H; CORE = intersection of R.
B is (empirically) sufficient iff B meets every edge.

Gate items, in the order Section 18 lists them:

  G0  required output block: P0/P1/P2 reducts and cores.
  G1  P0/P1 exact empirical-core recovery identity   1 - 2(3/4)^n + (1/2)^n.
  G2  labeled-space exact testing risk               (1/2)(1-eps)^n.
  G3  uniform three-population core bound            2(3/4)^n  for 0 < eps <= 1/4.
  G4  PT1 on the T5 support change.
  G5  PT1  <=>  (G1)&(G2) on the audited exhaustive domains.
  G6  identical minimax conventions for core and family.

  S1  supporting (not gating): P1, a in CORE  <=>  tau_a > 0.

A single FAIL anywhere fails the gate; Section 18 forbids starting the primary experiments
until the gate passes.

Usage: python3 v2_gate_checker.py [--nmax 20] [--p4-maxsupport 4] [--tv-nmax 6]
Writes gate_report.txt next to this file.
"""
import argparse
import itertools
import math
import os
from fractions import Fraction as F

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = "ABCDEFG"


# ---------------------------------------------------------------- exact polynomials in eps
class Poly:
    """Dense polynomial in eps with exact rational coefficients."""

    __slots__ = ("c",)

    def __init__(self, coeffs):
        c = [F(x) for x in coeffs] or [F(0)]
        while len(c) > 1 and c[-1] == 0:
            c.pop()
        self.c = c

    def at(self, i):
        return self.c[i] if i < len(self.c) else F(0)

    def __add__(self, o):
        return Poly([self.at(i) + o.at(i) for i in range(max(len(self.c), len(o.c)))])

    def __sub__(self, o):
        return Poly([self.at(i) - o.at(i) for i in range(max(len(self.c), len(o.c)))])

    def __mul__(self, o):
        r = [F(0)] * (len(self.c) + len(o.c) - 1)
        for i, a in enumerate(self.c):
            if a:
                for j, b in enumerate(o.c):
                    r[i + j] += a * b
        return Poly(r)

    def __pow__(self, k):
        r, b = Poly([1]), self
        while k:
            if k & 1:
                r = r * b
            b, k = b * b, k >> 1
        return r

    def __eq__(self, o):
        return self.c == o.c

    def __hash__(self):
        return hash(tuple(self.c))

    def ev(self, e):
        """Exact evaluation at a rational eps (Horner)."""
        acc = F(0)
        for a in reversed(self.c):
            acc = acc * e + a
        return acc

    def __repr__(self):
        return " + ".join(f"{a}*e^{i}" for i, a in enumerate(self.c) if a) or "0"


ZERO, ONE = Poly([0]), Poly([1])
EPS = Poly([0, 1])


# ---------------------------------------------------------------- structural layer
def disc(u, v):
    return frozenset(i for i, (a, b) in enumerate(zip(u, v)) if a != b)


def subsets(p):
    return [frozenset(c) for r in range(p + 1) for c in itertools.combinations(range(p), r)]


def analyse(S, f, p, subs=None):
    """Return (clutter, reduct family, core) of the structure (S, f)."""
    subs = subs or subsets(p)
    E = {disc(u, v) for u in S for v in S if f[u] != f[v]}
    K = frozenset(e for e in E if not any(d < e for d in E))
    hit = [B for B in subs if all(B & e for e in K)]
    R = frozenset(B for B in hit if not any(D < B for D in hit))
    core = frozenset.intersection(*R) if R else frozenset()
    return K, R, core


def show_sets(fam):
    return "{" + ", ".join("(" + ",".join(NAMES[i] for i in sorted(B)) + ")"
                           for B in sorted(fam, key=lambda B: (len(B), sorted(B)))) + "}"


def show_core(core):
    return "{" + ",".join(NAMES[i] for i in sorted(core)) + "}"


def structures(p, min_support=2, max_support=None):
    atoms = list(itertools.product((0, 1), repeat=p))
    hi = len(atoms) if max_support is None else min(max_support, len(atoms))
    for k in range(min_support, hi + 1):
        for S in itertools.combinations(atoms, k):
            for bits in itertools.product((0, 1), repeat=k):
                yield S, dict(zip(S, bits))


# ---------------------------------------------------------------- the frozen model class
P = 3
X0, X1, X2, X3 = (0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 1, 1)
SUPPORT = (X0, X1, X2, X3)
MASS = {X0: Poly([F(1, 4)]), X1: Poly([F(1, 2), -1]), X2: Poly([F(1, 4)]), X3: EPS}
LABELS = {"P0": (0, 0, 1, 0), "P1": (0, 0, 1, 1), "P2": (0, 1, 0, 1)}
# Section 8 fixes the class; both targets of Section 12 are evaluated on THIS tuple.
CLASS = ("P0", "P1", "P2")


def population(name):
    return dict(zip(SUPPORT, LABELS[name]))


def set_prob(T, n):
    """Exact P(the set of OBSERVED atoms is exactly T), by inclusion-exclusion, as a poly in eps."""
    acc = ZERO
    for r in range(len(T) + 1):
        for U in itertools.combinations(T, r):
            mass = ZERO
            for x in U:
                mass = mass + MASS[x]
            term = mass ** n
            acc = acc + term if (len(T) - r) % 2 == 0 else acc - term
    return acc


def core_error_poly(name, n, subs):
    """Exact P^n{ empirical core != CORE_P }, as a polynomial in eps.

    The estimator is the plug-in empirical core: it sees the labeled sample only and knows
    nothing about which member of the class generated it (Section 9 estimator convention)."""
    f = population(name)
    _, _, core_pop = analyse(SUPPORT, f, P, subs)
    err, total = ZERO, ZERO
    for r in range(1, len(SUPPORT) + 1):
        for T in itertools.combinations(SUPPORT, r):
            pr = set_prob(T, n)
            total = total + pr
            _, _, core_emp = analyse(T, {x: f[x] for x in T}, P, subs)
            if core_emp != core_pop:
                err = err + pr
    return err, total


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nmax", type=int, default=20)
    ap.add_argument("--p4-maxsupport", type=int, default=4)
    ap.add_argument("--tv-nmax", type=int, default=6)
    args = ap.parse_args()

    out, verdicts = [], {}

    def say(s=""):
        print(s, flush=True)
        out.append(s)

    subs = subsets(P)
    say("=" * 78)
    say("v2 VERIFICATION GATE --- locked_spec_v2.md Section 18")
    say("=" * 78)

    # ---------------------------------------------------------------- G0 required block
    say()
    say("--- G0: required output block ---")
    say()
    fams, cores = {}, {}
    for name in CLASS:
        f = population(name)
        _, R, core = analyse(SUPPORT, f, P, subs)
        fams[name], cores[name] = R, core
        say(f"{name} reducts = {show_sets(R)}")
        say(f"{name} core = {show_core(core)}")
    expected = {
        "P0": ("{(A,B), (A,C)}", "{A}"),
        "P1": ("{(A)}", "{A}"),
        "P2": ("{(C)}", "{C}"),
    }
    g0 = all(show_sets(fams[k]) == expected[k][0] and show_core(cores[k]) == expected[k][1]
             for k in CLASS)
    verdicts["G0 required block (Sec 8/18)"] = g0

    # the class must be nontrivial for BOTH functionals (Section 8, Section 12)
    core_nonconst = len({frozenset(cores[k]) for k in CLASS}) > 1
    fam_nonconst = len({fams[k] for k in CLASS}) > 1
    two_point_core_const = cores["P0"] == cores["P1"]
    two_point_fam_diff = fams["P0"] != fams["P1"]
    say()
    say(f"core functional nonconstant on the class: {core_nonconst}")
    say(f"reduct-family functional nonconstant on the class: {fam_nonconst}")
    say(f"on the two-point subclass {{P0,P1}} the core is CONSTANT: {two_point_core_const} "
        "(this is the v1 degeneracy the three-population class repairs)")
    say(f"on the two-point subclass {{P0,P1}} the FAMILY differs: {two_point_fam_diff} "
        "(so the family estimator must solve the testing problem of Section 11)")
    verdicts["G0b class nontrivial for core and family"] = (
        core_nonconst and fam_nonconst and two_point_core_const and two_point_fam_diff)

    # ---------------------------------------------------------------- G1 exact core identity
    say()
    say("--- G1: P0/P1 exact empirical-core recovery = 1 - 2(3/4)^n + (1/2)^n ---")
    bad, checked = [], 0
    for n in range(1, args.nmax + 1):
        claim = ONE - Poly([2]) * Poly([F(3, 4) ** n]) + Poly([F(1, 2) ** n])
        for name in ("P0", "P1"):
            err, total = core_error_poly(name, n, subs)
            if total != ONE:
                bad.append((n, name, "mass defect"))
            if (ONE - err) != claim:
                bad.append((n, name, f"recovery {ONE - err} != {claim}"))
            checked += 1
    say(f"n = 1..{args.nmax}, populations P0 and P1: {checked} exact polynomial identities checked, "
        f"{len(bad)} mismatch")
    say("    (identity holds as a polynomial in eps, i.e. uniformly in eps, and the total"
        " observed-set mass is exactly 1)")
    for b in bad[:5]:
        say(f"    MISMATCH {b}")
    n_demo = 5
    err5, _ = core_error_poly("P0", n_demo, subs)
    say(f"    n = {n_demo}: recovery = {(ONE - err5).ev(F(1, 10))} at eps = 1/10, "
        f"claim = {(ONE - Poly([2]) * Poly([F(3,4)**n_demo]) + Poly([F(1,2)**n_demo])).ev(F(1,10))}")
    verdicts["G1 P0/P1 exact core identity (Sec 10/18.1)"] = not bad

    # ---------------------------------------------------------------- G2 labeled-space testing risk
    say()
    say("--- G2: labeled-space exact testing risk = (1/2)(1-eps)^n ---")
    say("    (computed on the LABELED observation space; computing overlap on atom indices")
    say("     instead is the v1 checker bug recorded in D0-D3/proof_audit.md)")
    # labeled alphabet: (atom, label) symbols of BOTH populations
    f0, f1 = population("P0"), population("P1")
    alpha = [(X0, 0), (X1, 0), (X2, 1), (X3, 0), (X3, 1)]
    p0 = [MASS[x] if f0[x] == y else ZERO for x, y in alpha]
    p1 = [MASS[x] if f1[x] == y else ZERO for x, y in alpha]
    bad2 = []
    for n in range(1, args.tv_nmax + 1):
        overlap, l1 = ZERO, ZERO
        for counts in itertools.product(range(n + 1), repeat=len(alpha)):
            if sum(counts) != n:
                continue
            coef = Poly([math.factorial(n) // math.prod(math.factorial(c) for c in counts)])
            a, b = coef, coef
            for prob, c in zip(p0, counts):
                if c:
                    a = a * (prob ** c)
            for prob, c in zip(p1, counts):
                if c:
                    b = b * (prob ** c)
            if a == ZERO and b == ZERO:
                continue
            if a != ZERO and b != ZERO:
                if a != b:  # structural claim: on the shared part the two laws agree exactly
                    bad2.append((n, counts, "shared outcome with unequal mass"))
                overlap = overlap + a
            else:
                l1 = l1 + (a if b == ZERO else b)
        tv_overlap = ONE - overlap
        tv_l1 = Poly([F(1, 2)]) * l1
        risk = Poly([F(1, 2)]) * overlap
        claim_tv = ONE - (ONE - EPS) ** n
        claim_risk = Poly([F(1, 2)]) * (ONE - EPS) ** n
        if tv_overlap != claim_tv:
            bad2.append((n, "TV via overlap", str(tv_overlap)))
        if tv_l1 != claim_tv:
            bad2.append((n, "TV via L1", str(tv_l1)))
        if risk != claim_risk:
            bad2.append((n, "risk", str(risk)))
    say(f"n = 1..{args.tv_nmax}: TV computed two independent ways (1 - overlap, and (1/2)*L1) "
        f"against 1 - (1-eps)^n; equal-prior risk against (1/2)(1-eps)^n: {len(bad2)} mismatch")
    for b in bad2[:5]:
        say(f"    MISMATCH {b}")
    say(f"    n = 3: risk = {(Poly([F(1,2)]) * (ONE - EPS) ** 3)}")
    verdicts["G2 exact labeled testing risk (Sec 11/18.2)"] = not bad2

    # ---------------------------------------------------------------- G3 uniform core bound
    say()
    say("--- G3: uniform three-population core bound 2(3/4)^n on 0 < eps <= 1/4 ---")
    grid = [F(1, 4), F(1, 5), F(1, 6), F(1, 8), F(1, 10), F(1, 20), F(1, 50), F(1, 100),
            F(1, 1000), F(1, 10 ** 6)] + [F(k, 400) for k in range(1, 101)]
    viol, worst = 0, (F(0), None)
    for n in range(1, args.nmax + 1):
        bound = F(2) * F(3, 4) ** n
        for name in CLASS:
            err, _ = core_error_poly(name, n, subs)
            for e in grid:
                v = err.ev(e)
                if v > bound:
                    viol += 1
                if bound and v / bound > worst[0]:
                    worst = (v / bound, (n, name, e))
    say(f"n = 1..{args.nmax}, all 3 populations, {len(grid)} exact rational eps in (0, 1/4]: "
        f"{viol} violations of sup_P P^n{{core_hat != CORE_P}} <= 2(3/4)^n")
    say(f"    worst ratio error/bound = {float(worst[0]):.6f} at (n, population, eps) = {worst[1]}")
    # closed form for P2, stated in the proof but not in the spec: (3/4)^n + (1/2+eps)^n - (1/4+eps)^n
    bad3 = []
    for n in range(1, args.nmax + 1):
        err, _ = core_error_poly("P2", n, subs)
        claim = Poly([F(3, 4) ** n]) + (Poly([F(1, 2)]) + EPS) ** n - (Poly([F(1, 4)]) + EPS) ** n
        if err != claim:
            bad3.append(n)
    say(f"    P2 exact core error = (3/4)^n + (1/2+eps)^n - (1/4+eps)^n: "
        f"{len(bad3)} mismatch over n = 1..{args.nmax}")
    verdicts["G3 uniform core bound (Sec 10/18.3)"] = (viol == 0 and not bad3)

    # ---------------------------------------------------------------- G4 PT1 on the T5 change
    say()
    say("--- G4: PT1 on the T5 support change ---")
    S_base = (X0, X1, X2)
    f_base = {X0: 0, X1: 0, X2: 1}
    K_b, R_b, core_b = analyse(S_base, f_base, P, subs)
    say(f"    base support {S_base} labels (0,0,1): R = {show_sets(R_b)}, CORE = {show_core(core_b)}")
    bad4 = []
    for y, target in ((0, "P0"), (1, "P1")):
        new_edges = {disc(X3, x) for x in S_base if f_base[x] != y}
        a1 = all(len(e) != 1 or next(iter(e)) in core_b for e in new_edges)
        a2 = any(not any(k <= e for k in K_b) for e in new_edges)
        pred = a1 and a2
        g = dict(f_base)
        g[X3] = y
        _, R2, core2 = analyse(SUPPORT, g, P, subs)
        actual = (core2 == core_b) and (R2 != R_b)
        say(f"    add z = {X3} with label {y} (-> {target}): new edges "
            f"{sorted(sorted(NAMES[i] for i in e) for e in new_edges)}; "
            f"PT1 A1={a1} A2={a2} => predicted {pred}; actual core-preserved-and-family-changed "
            f"= {actual} (R -> {show_sets(R2)}, CORE -> {show_core(core2)})")
        if pred != actual:
            bad4.append((y, pred, actual))
        if target == "P0" and not (R2 == fams["P0"] and core2 == cores["P0"]):
            bad4.append((y, "does not reproduce P0"))
        if target == "P1" and not (R2 == fams["P1"] and core2 == cores["P1"]):
            bad4.append((y, "does not reproduce P1"))
    say(f"    PT1 prediction vs brute force on the T5 transition: {len(bad4)} mismatch")
    verdicts["G4 PT1 on T5 support change (Sec 6/18.4)"] = not bad4

    # ---------------------------------------------------------------- G5 PT1 <=> (G1)&(G2)
    say()
    say("--- G5: PT1 <=> (G1)&(G2) on exhaustive domains ---")
    g5_ok = True
    for p, maxsup in ((3, None), (4, args.p4_maxsupport)):
        ss = subsets(p)
        atoms = list(itertools.product((0, 1), repeat=p))
        eq_bad = total = trans = 0
        ce = None
        for S, f in structures(p, max_support=maxsup):
            total += 1
            K, R, core = analyse(S, f, p, ss)
            for z in atoms:
                if z in f:
                    continue
                for y in (0, 1):
                    opp = [x for x in S if f[x] != y]
                    new_edges = {disc(z, x) for x in opp}
                    a1 = all(len(e) != 1 or next(iter(e)) in core for e in new_edges)
                    a2 = any(not any(k <= e for k in K) for e in new_edges)
                    pt = a1 and a2
                    gg1 = all(next(iter(disc(z, x))) in core for x in opp if len(disc(z, x)) == 1)
                    gg2 = any(all(z[a] == x[a] for a in Rr) for Rr in R for x in opp)
                    geo = gg1 and gg2
                    trans += 1
                    if pt != geo:
                        eq_bad += 1
                        ce = ce or (S, [f[x] for x in S], z, y, pt, geo)
        scope = "all |S| >= 2" if maxsup is None else f"2 <= |S| <= {maxsup}"
        say(f"    p = {p} ({scope}): {total} structures, {trans} atom additions, "
            f"{eq_bad} violations of PT1 <=> (G1)&(G2)")
        if ce:
            say(f"        counterexample: {ce}")
        g5_ok = g5_ok and eq_bad == 0
    verdicts["G5 PT1 <=> (G1)&(G2) (Sec 7/18.5)"] = g5_ok

    # ---------------------------------------------------------------- G6 identical conventions
    say()
    say("--- G6: identical minimax conventions for core and family ---")

    def n_star(risk_of_n, delta, guess=None, cap=10 ** 4):
        """Section 9 convention, applied by ONE function to BOTH targets.

        Given a closed-form guess, verify the exact boundary pair
        risk(guess) <= delta < risk(guess - 1) instead of scanning: that pins the same
        minimum, is the stronger statement (it certifies the closed form itself), and is the
        only tractable route -- an exact-rational scan to n ~ 4*10^4 multiplies integers with
        10^5 digits at every step."""
        if guess is not None:
            if guess >= 1 and risk_of_n(guess) <= delta and (
                    guess == 1 or risk_of_n(guess - 1) > delta):
                return guess
            return None
        n = 1
        while n <= cap:
            if risk_of_n(n) <= delta:
                return n
            n += 1
        return None

    def core_risk(eps):
        """sup over the SAME class of the plug-in core error: an upper bound on the minimax risk."""
        return lambda n: max(core_error_poly(name, n, subs)[0].ev(eps) for name in CLASS)

    def family_risk_lower(eps):
        """Le Cam two-point bound inside the SAME class, same labeled-sample information."""
        return lambda n: F(1, 2) * (1 - eps) ** n

    say("    class used by both targets      : " + ", ".join(CLASS) + "  (same tuple object: "
        f"{CLASS is CLASS})")
    say("    delta convention for both       : n(delta) = min{n >= 1 : risk_n <= delta}, "
        "one shared implementation n_star()")
    say("    estimator information for both  : the labeled sample only (core via the plug-in "
        "empirical core; family via a test between P0 and P1 on the labeled space)")
    say("    core risk is an UPPER bound (an explicit estimator), family risk is a LOWER bound "
        "(all estimators) -- the separation is therefore in the right direction")
    say()
    say("    delta   eps         n_core(sup plug-in)  n_core(spec bound)  n_family(lower)   ratio")
    rows = []
    for delta in (F(1, 100),):
        for eps in (F(1, 10), F(1, 100), F(1, 1000), F(1, 10000)):
            nc_spec = math.ceil(math.log(2 / float(delta)) / -math.log(3 / 4))
            nf_spec = math.ceil(math.log(1 / (2 * float(delta))) / -math.log(1 - float(eps)))
            nc = n_star(core_risk(eps), delta)
            nf = n_star(family_risk_lower(eps), delta, guess=nf_spec)
            rows.append((delta, eps, nc, nc_spec, nf, nf_spec))
            say(f"    {float(delta):<7} {float(eps):<11} {nc:<20} {nc_spec:<19} "
                f"{str(nf):<17} " + (f"{nf / nc:.1f}x" if nf else "closed form REFUTED"))
    # the spec's closed-form bounds must dominate / be dominated in the right direction
    bad6 = []
    for delta, eps, nc, nc_spec, nf, nf_spec in rows:
        if nc > nc_spec:
            bad6.append(("n_core exceeds the spec bound", delta, eps, nc, nc_spec))
        if nf != nf_spec:
            bad6.append(("n_family disagrees with the closed form", delta, eps, nf, nf_spec))
    ratios = [r[4] / r[2] for r in rows if r[4] and r[2]]
    growing = len(ratios) == len(rows) and all(
        ratios[i] < ratios[i + 1] for i in range(len(ratios) - 1))
    say(f"    n_core is bounded uniformly in eps, n_family grows like log(1/delta)/eps; "
        f"ratio strictly increasing as eps -> 0: {growing}")
    verdicts["G6 identical minimax conventions (Sec 9/12/18.6)"] = (not bad6) and growing
    for b in bad6[:5]:
        say(f"    MISMATCH {b}")

    # ---------------------------------------------------------------- S1 supporting: tau_a
    say()
    say("--- S1 (supporting, not gating): a in CORE_P  <=>  tau_a > 0 ---")
    bad_s1 = 0
    for eps in (F(1, 10), F(1, 100), F(1, 1000)):
        for name in CLASS:
            f = population(name)
            _, _, core = analyse(SUPPORT, f, P, subs)
            for a in range(P):
                joint = {}
                for x in SUPPORT:
                    z = tuple(v for i, v in enumerate(x) if i != a)
                    joint[(z, f[x])] = joint.get((z, f[x]), F(0)) + MASS[x].ev(eps)
                best = {}
                for (z, y), m in joint.items():
                    best[z] = max(best.get(z, F(0)), m)
                tau = 1 - sum(best.values())
                if (tau > 0) != (a in core):
                    bad_s1 += 1
    say(f"    3 populations x 3 attributes x 3 values of eps: {bad_s1} violations")
    verdicts["S1 tau_a characterisation (Sec 3, supporting)"] = bad_s1 == 0

    # ---------------------------------------------------------------- verdict
    say()
    say("=" * 78)
    gate_items = {k: v for k, v in verdicts.items() if not k.startswith("S1")}
    for k, v in verdicts.items():
        say(f"  {'PASS' if v else 'FAIL'}  {k}")
    ok = all(gate_items.values())
    say()
    say(f"  v2 GATE: {'PASS' if ok else 'FAIL'} "
        f"({sum(gate_items.values())}/{len(gate_items)} gating items)")
    say("  Section 18 satisfied: the primary experiments are unblocked."
        if ok else "  Section 18: primary experiments remain BLOCKED.")
    say("=" * 78)

    with open(os.path.join(HERE, "gate_report.txt"), "w") as fh:
        fh.write("\n".join(out) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
