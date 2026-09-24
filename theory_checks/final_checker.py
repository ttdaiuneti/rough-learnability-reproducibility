#!/usr/bin/env python3
"""Final A-ceiling checker. Enumeration only; no heuristic search certifies anything.

Structure = (S, f), S subset of {0,1}^p, f: S -> {0,1}. Objects (see ../D5-hypergraph-theory/):
  edges D(u,v) for bichromatic pairs; clutter = minimal edges; R = minimal transversals; CORE = intersection.

Conjectures stated BEFORE running (see problem_definition.md, projection_theory.md):

 CJ1 (proved, checked): PT1(z,y) <=> (G1) and (G2), where
      (G1) every x in S with f(x) != y at Hamming distance 1 from z differs in a CORE attribute;
      (G2) there are R in R_P and x in S with f(x) != y such that pi_R(z) = pi_R(x).
 CJ2 (proved sufficient, checked for FP=0): if some R in R_P has 2^(p-|R|) > |S|*(p+1) then P is vulnerable.
 CJ3 (conjectured FALSE, expect a counterexample): |R_P| = 1 and CORE sufficient  =>  immune.
 CJ4 (exploratory): cluster immune structures with S strictly inside the cube by candidate invariants
      (|S|, |R|, min |R|, core size, core sufficient, projection-completeness of every reduct).

Vulnerability is a property of the labelled support only (masses cancel), see problem_definition.md.

Usage: python3 final_checker.py [--p4-maxsupport 5]
Writes geometric_translation_report.txt next to this file.
"""
import argparse
import itertools
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = "ABCDEFG"


def disc(u, v):
    return frozenset(i for i, (a, b) in enumerate(zip(u, v)) if a != b)


def subsets(p):
    return [frozenset(c) for r in range(p + 1) for c in itertools.combinations(range(p), r)]


def analyse(S, f, p, subs):
    E = {disc(u, v) for u in S for v in S if f[u] != f[v]}
    K = frozenset(e for e in E if not any(d < e for d in E))
    hit = [B for B in subs if all(B & e for e in K)]
    R = frozenset(B for B in hit if not any(D < B for D in hit))
    core = frozenset.intersection(*R) if R else frozenset()
    return K, R, core


def pt1(K, core, new_edges):
    """D5 local condition: (A1) no new singleton outside the core; (A2) a new edge containing no old edge."""
    a1 = all(len(e) != 1 or next(iter(e)) in core for e in new_edges)
    a2 = any(not any(k <= e for k in K) for e in new_edges)
    return a1 and a2


def geometric(S, f, p, R, core, z, y):
    """(G1) and (G2) in support/projection language."""
    opp = [x for x in S if f[x] != y]
    g1 = all(next(iter(disc(z, x))) in core for x in opp if len(disc(z, x)) == 1)
    g2 = any(all(z[a] == x[a] for a in Rr) for Rr in R for x in opp)
    return g1 and g2


def structures(p, min_support=2, max_support=None):
    atoms = list(itertools.product((0, 1), repeat=p))
    hi = len(atoms) if max_support is None else min(max_support, len(atoms))
    for k in range(min_support, hi + 1):
        for S in itertools.combinations(atoms, k):
            for bits in itertools.product((0, 1), repeat=k):
                yield S, dict(zip(S, bits))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p4-maxsupport", type=int, default=5)
    args = ap.parse_args()
    out = []

    def say(s=""):
        print(s, flush=True)
        out.append(s)

    for p, maxsup in ((3, None), (4, args.p4_maxsupport)):
        subs = subsets(p)
        atoms = list(itertools.product((0, 1), repeat=p))
        eq_bad = 0
        vuln = total = 0
        cj2_TP = cj2_FP = 0
        cj3_counter = None
        immune_profiles = Counter()
        immune_examples = {}
        smallest_eq_ce = None
        prof_split = {}
        cj6_total = cj6_bad = 0
        from collections import defaultdict
        prof_split = defaultdict(lambda: [0, 0])
        for S, f in structures(p, max_support=maxsup):
            total += 1
            K, R, core = analyse(S, f, p, subs)
            is_vuln = False
            for z in atoms:
                if z in f:
                    continue
                for y in (0, 1):
                    new_edges = {disc(z, x) for x in S if f[x] != y}
                    a = pt1(K, core, new_edges)
                    b = geometric(S, f, p, R, core, z, y)
                    if a != b:
                        eq_bad += 1
                        smallest_eq_ce = smallest_eq_ce or (S, [f[x] for x in S], z, y, a, b)
                    is_vuln = is_vuln or a
            vuln += is_vuln
            # CJ2: hole-free sufficient condition
            sparse = any(2 ** (p - len(Rr)) > len(S) * (p + 1) for Rr in R)
            if sparse:
                if is_vuln:
                    cj2_TP += 1
                else:
                    cj2_FP += 1
            # CJ3: unique reduct and sufficient core => immune?
            core_suff = all(core & e for e in K) if K else True
            if len(R) == 1 and core_suff and is_vuln and cj3_counter is None:
                cj3_counter = (S, [f[x] for x in S])
            # CJ4/CJ5: profile EVERY structure, so that we can test whether the invariant vector
            # separates immune from vulnerable (CJ5). Profiling only the immune ones would let us
            # read a non-existent law out of the data.
            projfull = all(len({tuple(x[a] for a in sorted(Rr)) for x in S}) == 2 ** len(Rr) for Rr in R)
            prof = (len(S), len(R), min((len(Rr) for Rr in R), default=0), len(core), core_suff, projfull)
            prof_split[prof][0 if is_vuln else 1] += 1
            if not is_vuln and len(S) < len(atoms):
                immune_profiles[prof] += 1
                immune_examples.setdefault((len(S), len(R), projfull), (S, [f[x] for x in S]))
            # CJ6: is the only reduct the whole attribute set?  (proved sufficient for immunity)
            if R == frozenset({frozenset(range(p))}):
                cj6_total += 1
                if is_vuln:
                    cj6_bad += 1
        scope = "all |S| >= 2" if maxsup is None else f"2 <= |S| <= {maxsup}"
        say(f"=== p = {p} ({scope}): {total} structures ===")
        say(f"CJ1 PT1 <=> (G1)&(G2): {eq_bad} violations")
        if smallest_eq_ce:
            say(f"    counterexample: {smallest_eq_ce}")
        say(f"vulnerable structures (P in U): {vuln} ({100 * vuln / total:.1f}%), immune: {total - vuln}")
        say(f"CJ2 sparse-cylinder sufficient condition: TP={cj2_TP} FP={cj2_FP} (FP must be 0)")
        say(f"CJ3 'unique reduct + sufficient core => immune': "
            + ("FALSIFIED, counterexample " + str(cj3_counter) if cj3_counter else "no counterexample found"))
        mixed = {pr: c for pr, c in prof_split.items() if c[0] > 0 and c[1] > 0}
        say(f"CJ5 does the invariant vector separate vulnerable from immune? "
            f"profiles={len(prof_split)}, MIXED profiles (both outcomes)={len(mixed)} "
            f"-> {'NO separation' if mixed else 'separates'}")
        for pr, c in sorted(mixed.items())[:6]:
            say(f"      mixed profile {pr}: vulnerable={c[0]} immune={c[1]}")
        say(f"CJ6 'only reduct is the full attribute set => immune': {cj6_total} such structures, "
            f"{cj6_bad} violations")
        say(f"CJ4 immune structures with incomplete support: {sum(immune_profiles.values())}")
        say("    profile (|S|, |R|, min|R|, |core|, core_sufficient, all_reduct_projections_full) -> count")
        for prof, cnt in sorted(immune_profiles.items())[:14]:
            say(f"      {prof} -> {cnt}")
        say()

    with open(os.path.join(HERE, "geometric_translation_report.txt"), "w") as fh:
        fh.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
