"""POST-HOC check (amendment A1 of theory_spec.md, added after the smoke run): Proposition F4''.

For every labeled support with CORE_P nonempty and every atom z with S \\ {z} nonempty:
  LHS: {z} is a core-preserving cover of G_e for some e in C_P
  RHS: adding z to Q = P(.|S\\{z}) satisfies manuscript Prop. 10's conditions
       (C1) every new singleton edge {b} = D(z,x) has b in CORE_Q,
       (C2) some new edge D(z,x) contains no edge of C_Q,
       and CORE_Q = CORE_P.
Also records whether (C1) alone implies CORE_Q = CORE_P (Prop. 10's core statement).
Structures only; masses play no role. Families recomputed literally.
"""
import itertools
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from k1_k3_checker import (Support, clutter_of, core_of, disc, family_literal,  # noqa: E402
                           supports)


def check(p, labels, kmin, kmax):
    st = dict(p=p, labels=list(labels), sizes=[kmin, kmax], cases=0, lhs_true=0, mismatches=0,
              c1_without_core_equal=0, examples=[])
    for S, lab in supports(p, labels, kmin, kmax):
        sup = Support(S, lab, p)
        if not sup.core:
            continue
        for zi, z in enumerate(sup.atoms):
            rest = [x for x in sup.atoms if x != z]
            if not rest:
                continue
            st["cases"] += 1
            K = 1 << zi
            lhs = False
            for e in sup.clutter:
                pairs = sup.pairs[e]
                if not all(i == zi or j == zi for i, j in pairs):
                    continue
                cp = all(any(i != zi and j != zi for i, j in sup.pairs[frozenset([b])]) for b in sup.core)
                if cp:
                    lhs = True
                    break
            famQ = family_literal(rest, lab, p)
            coreQ = core_of(famQ)
            clQ = clutter_of(rest, lab)
            new = [disc(z, x) for x in rest if lab[x] != lab[z]]
            c1 = all(len(d) != 1 or next(iter(d)) in coreQ for d in new)
            c2 = any(not any(f <= d for f in clQ) for d in new)
            rhs = c1 and c2 and coreQ == sup.core
            if c1 and coreQ != sup.core:
                st["c1_without_core_equal"] += 1
            st["lhs_true"] += lhs
            if lhs != rhs:
                st["mismatches"] += 1
                if len(st["examples"]) < 10:
                    st["examples"].append(dict(S=S, lab=[lab[x] for x in S], z=z, lhs=lhs,
                                               c1=c1, c2=c2, coreQ=sorted(coreQ), coreP=sorted(sup.core)))
    return st


def main():
    out = []
    for cfg in [(3, (0, 1), 2, 5), (4, (0, 1), 2, 4), (3, (0, 1, 2), 2, 4)]:
        st = check(*cfg)
        print(json.dumps({k: v for k, v in st.items() if k != "examples"}), flush=True)
        out.append(st)
    verdict = "PASS" if all(s["mismatches"] == 0 for s in out) else "FAIL"
    json.dump(dict(configs=out, verdict=verdict), open(os.path.join(HERE, "f4pp_results.json"), "w"),
              indent=1, default=str)
    print("F4'' VERDICT", verdict)


if __name__ == "__main__":
    main()
