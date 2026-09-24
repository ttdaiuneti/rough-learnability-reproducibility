#!/usr/bin/env python3
"""Exhaustive checker for the D5 hypergraph stage. Enumeration only; no heuristic search anywhere.

A structure is (S, f): S a subset of {0,1}^p, f: S -> {0,1}. Discernibility edges D(u,v) = {a : u_a != v_a}
for bichromatic pairs; H = set of edges; C = min_subset(H) the clutter; R = minimal transversals; CORE = the
intersection of R.

Conjectures tested (see ../D5-hypergraph-theory/*.md):
  V1 sufficiency <=> transversal of H
  V2 Tr_min(H) = Tr_min(C)
  V3 CORE = singleton edges of H = singleton edges of C
  V4 blocker duality b(b(C)) = C, and R_P = R_Q  <=>  C_P = C_Q
  V5 realizability: every clutter on C arises from an explicit 2-class binary construction
  V6 full-cube rigidity: supp = {0,1}^p  =>  |R| = 1 and R = {CORE}
  V7 ADDITION THEOREM (main conjecture), TP/TN/FP/FN over all single-atom additions
  V8 is the clutter a sufficient statistic for RELABEL perturbations?  (expected: no)
  V9 three-population repair: labels (0,1,0,1) give R = {{C}}, CORE = {C}

Usage: python3 hypergraph_checker.py [--p4-maxsupport 4]
Writes atom_addition_report.txt next to this file.
"""
import argparse
import itertools
import os

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = "ABCDEFG"


def disc(u, v):
    return frozenset(i for i, (a, b) in enumerate(zip(u, v)) if a != b)


def edges(S, f):
    return {disc(u, v) for u in S for v in S if f[u] != f[v]}


def clutter(E):
    return frozenset(e for e in E if not any(d < e for d in E))


def subsets(p):
    return [frozenset(c) for r in range(p + 1) for c in itertools.combinations(range(p), r)]


def transversals_min(E, p):
    hit = [B for B in subsets(p) if all(B & e for e in E)]
    return frozenset(B for B in hit if not any(D < B for D in hit))


def core_of(R):
    return frozenset.intersection(*R) if R else frozenset()


def singletons(E):
    return frozenset(next(iter(e)) for e in E if len(e) == 1)


def analyse(S, f, p):
    E = edges(S, f)
    K = clutter(E)
    R = transversals_min(E, p)
    return E, K, R, core_of(R)


def structures(p, min_support=2, max_support=None):
    atoms = list(itertools.product((0, 1), repeat=p))
    hi = len(atoms) if max_support is None else min(max_support, len(atoms))
    for k in range(min_support, hi + 1):
        for S in itertools.combinations(atoms, k):
            for bits in itertools.product((0, 1), repeat=k):
                yield S, dict(zip(S, bits))


def antichains(p):
    """All clutters (antichains of nonempty subsets) on p attributes, by enumeration."""
    nonempty = [s for s in subsets(p) if s]
    out = []
    for r in range(len(nonempty) + 1):
        for comb in itertools.combinations(nonempty, r):
            if all(not (a < b) and not (b < a) for a, b in itertools.combinations(comb, 2)):
                out.append(frozenset(comb))
    return out


def realize(K, p):
    """Explicit construction: base point 0 with label 0, and for every edge e of K the indicator of e
    with label 1. Returns (S, f)."""
    z = tuple(0 for _ in range(p))
    S = [z] + [tuple(1 if i in e else 0 for i in range(p)) for e in sorted(K, key=sorted)]
    f = {z: 0}
    for x in S[1:]:
        f[x] = 1
    return tuple(S), f


def addition_prediction(K_P, core_P, new_edges):
    """Conditions of the addition theorem, computed from the OLD clutter and the new discernibility sets only.
    (1) every new singleton edge is already a core attribute;
    (2) some new edge contains no edge of the old clutter."""
    c1 = all((len(e) != 1) or (next(iter(e)) in core_P) for e in new_edges)
    c2 = any(not any(k <= e for k in K_P) for e in new_edges)
    return c1 and c2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p4-maxsupport", type=int, default=4)
    args = ap.parse_args()
    out = []

    def say(s=""):
        print(s, flush=True)
        out.append(s)

    # ------------------------------------------------------------------ V1-V3, p = 3 exhaustive
    p = 3
    n = v1 = v2 = v3 = 0
    for S, f in structures(p):
        n += 1
        E, K, R, cp = analyse(S, f, p)
        for B in subsets(p):
            suff = all(not (f[u] != f[v] and not (B & disc(u, v))) for u in S for v in S)
            if suff != all(B & e for e in E):
                v1 += 1
        if transversals_min(E, p) != transversals_min(K, p):
            v2 += 1
        if not (cp == singletons(E) == singletons(K)):
            v3 += 1
    say(f"=== p = 3, all {n} structures with |S| >= 2 ===")
    say(f"V1 sufficiency <=> transversal: {v1} violations")
    say(f"V2 Tr_min(H) = Tr_min(clutter): {v2} violations")
    say(f"V3 CORE = singleton edges (of H and of clutter): {v3} violations")

    # ------------------------------------------------------------------ V4 blocker duality
    for pp in (3, 4):
        acs = antichains(pp)
        bad_dual = sum(1 for K in acs if K and transversals_min(transversals_min(K, pp), pp) != K)
        say(f"V4 blocker duality b(b(C)) = C on all {len(acs)} clutters, p = {pp}: {bad_dual} violations")

    # R_P = R_Q  <=>  C_P = C_Q, exhaustively over structures
    seen = {}
    bad_iff = 0
    for S, f in structures(3):
        E, K, R, cp = analyse(S, f, 3)
        if R in seen and seen[R] != K:
            bad_iff += 1
        seen.setdefault(R, K)
    say(f"V4 R determines the clutter (p = 3): {bad_iff} violations")

    # ------------------------------------------------------------------ V5 realizability
    for pp in (3, 4):
        acs = [K for K in antichains(pp) if K]
        bad = 0
        for K in acs:
            S, f = realize(K, pp)
            _, K2, _, _ = analyse(S, f, pp)
            if K2 != K:
                bad += 1
        say(f"V5 every clutter realizable by the explicit 2-class binary construction, p = {pp}: "
            f"{len(acs)} clutters, {bad} failures")

    # ------------------------------------------------------------------ V6 full-cube rigidity
    for pp in (3, 4):
        atoms = list(itertools.product((0, 1), repeat=pp))
        bad = nonconst = 0
        for bits in itertools.product((0, 1), repeat=len(atoms)):
            f = dict(zip(atoms, bits))
            if len(set(bits)) == 1:
                continue
            nonconst += 1
            _, _, R, cp = analyse(atoms, f, pp)
            relevant = frozenset(a for a in range(pp)
                                 if any(f[x] != f[tuple(1 - x[i] if i == a else x[i] for i in range(pp))]
                                        for x in atoms))
            if not (len(R) == 1 and R == frozenset({cp}) and cp == relevant):
                bad += 1
        say(f"V6 full-cube rigidity (|R| = 1, R = {{CORE}} = relevant variables), p = {pp}: "
            f"{nonconst} non-constant functions, {bad} violations")

    # ------------------------------------------------------------------ V7 addition theorem
    for pp, maxsup in ((3, None), (4, args.p4_maxsupport)):
        TP = TN = FP = FN = 0
        ce_fp = ce_fn = None
        atoms = list(itertools.product((0, 1), repeat=pp))
        for S, f in structures(pp, max_support=maxsup):
            E, K, R, cp = analyse(S, f, pp)
            for z in atoms:
                if z in f:
                    continue
                for y in (0, 1):
                    new_edges = {disc(z, x) for x in S if f[x] != y}
                    pred = addition_prediction(K, cp, new_edges)
                    g = dict(f)
                    g[z] = y
                    S2 = tuple(list(S) + [z])
                    _, _, R2, cp2 = analyse(S2, g, pp)
                    actual = (cp2 == cp) and (R2 != R)
                    if pred and actual:
                        TP += 1
                    elif not pred and not actual:
                        TN += 1
                    elif pred and not actual:
                        FP += 1
                        ce_fp = ce_fp or (S, [f[x] for x in S], z, y)
                    else:
                        FN += 1
                        ce_fn = ce_fn or (S, [f[x] for x in S], z, y)
        scope = "all |S| >= 2" if maxsup is None else f"2 <= |S| <= {maxsup}"
        say(f"V7 ADDITION THEOREM, p = {pp} ({scope}): TP={TP} TN={TN} FP={FP} FN={FN}")
        if ce_fp:
            say(f"    smallest false positive: support={ce_fp[0]} labels={ce_fp[1]} z={ce_fp[2]} y={ce_fp[3]}")
        if ce_fn:
            say(f"    smallest false negative: support={ce_fn[0]} labels={ce_fn[1]} z={ce_fn[2]} y={ce_fn[3]}")

    # ------------------------------------------------------------------ V8 clutter as sufficient statistic
    # Two structures with the SAME clutter; relabel "the same" atom-role and see whether the answer agrees.
    by_clutter = {}
    for S, f in structures(3):
        E, K, R, cp = analyse(S, f, 3)
        by_clutter.setdefault(K, []).append((S, dict(f), R, cp))
    disagree = 0
    example = None
    for K, items in by_clutter.items():
        if len(items) < 2:
            continue
        sigs = []
        for S, f, R, cp in items[:40]:
            outcomes = set()
            for zz in S:
                g = dict(f)
                g[zz] = 1 - g[zz]
                _, _, R2, cp2 = analyse(S, g, 3)
                outcomes.add((cp2 == cp, R2 != R))
            sigs.append(frozenset(outcomes))
        if len(set(sigs)) > 1:
            disagree += 1
            example = example or (K, [(it[0], it[1]) for it in items[:2]])
    say(f"V8 clutters (p = 3) whose relabel-outcome signature is NOT determined by the clutter: {disagree} "
        f"of {sum(1 for v in by_clutter.values() if len(v) >= 2)} multi-instance clutters")
    if example:
        K, two = example
        say(f"    example clutter {sorted(sorted(NAMES[i] for i in e) for e in K)}: "
            f"support {two[0][0]} vs {two[1][0]}")

    # ------------------------------------------------------------------ V9 three-population repair
    say()
    say("=== V9: three-population class ===")
    S = ((0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 1, 1))
    for name, labels in (("P0", (0, 0, 1, 0)), ("P1", (0, 0, 1, 1)), ("P2", (0, 1, 0, 1))):
        f = dict(zip(S, labels))
        E, K, R, cp = analyse(S, f, 3)
        say(f"  {name} labels={labels}: clutter={sorted(sorted(NAMES[i] for i in e) for e in K)} "
            f"R={sorted(sorted(NAMES[i] for i in B) for B in R)} CORE={sorted(NAMES[i] for i in cp)}")

    with open(os.path.join(HERE, "atom_addition_report.txt"), "w") as fh:
        fh.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
