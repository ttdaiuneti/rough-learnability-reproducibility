"""Independent exact check of the proposed Theorem 14 family upper bound.

No project checker imported. Families come from literal sufficiency; sample laws
come from a dynamic programme on observed subsets. This is post-hoc verification,
not a pre-registered experiment or a substitute for the proof.
"""
from fractions import Fraction as F
from itertools import combinations, product
from pathlib import Path
import json
import random

def structure(xs, ys, p, mask):
    sufficient = []
    for b in range(1 << p):
        seen = {}
        for i, x in enumerate(xs):
            if not mask >> i & 1:
                continue
            z = tuple(x[j] for j in range(p) if b >> j & 1)
            if z in seen and seen[z] != ys[i]:
                break
            seen[z] = ys[i]
        else:
            sufficient.append(b)
    fam = frozenset(b for b in sufficient if not any(c != b and c & b == c for c in sufficient))
    core = (1 << p) - 1
    for b in fam:
        core &= b
    return fam, core

def observed_law(weights, n):
    law = {0: F(1)}
    for _ in range(n):
        nxt = {}
        for mask, prob in law.items():
            for i, w in enumerate(weights):
                if w:
                    z = mask | (1 << i)
                    nxt[z] = nxt.get(z, F(0)) + prob * w
        law = nxt
    return law

def main():
    rng = random.Random(20260915)
    count = dict(supports=0, mass_assignments=0, separating_classes=0, risk_checks=0,
                 deletion_radius_checks=0, core_formula_checks=0)
    for p, q, max_k in [(3, 2, 4), (3, 3, 3), (4, 2, 3)]:
        for k in range(2, max_k + 1):
            full = (1 << k) - 1
            for xs in combinations(list(product((0, 1), repeat=p)), k):
                for ys in product(range(q), repeat=k):
                    if len(set(ys)) < 2:
                        continue
                    count['supports'] += 1
                    targets = {u: structure(xs, ys, p, full ^ u) for u in range(full)}
                    fam, core = targets[0]
                    for raw in [[1] * k, [rng.randint(1, 20) for _ in range(k)]]:
                        ws = [F(w, sum(raw)) for w in raw]
                        count['mass_assignments'] += 1
                        mass = {u: sum((ws[i] for i in range(k) if u >> i & 1), F(0)) for u in range(full)}
                        m = min(mass[u] for u in targets if targets[u][0] != fam)
                        edges = {sum(1 << j for j in range(p) if xs[i][j] != xs[l][j])
                                 for i, l in combinations(range(k), 2) if ys[i] != ys[l]}
                        clutter = [e for e in edges if not any(d != e and d & e == d for d in edges)]
                        covers = {e: [u for u in range(full) if all(u >> i & 1 or u >> l & 1
                                  for i, l in combinations(range(k), 2)
                                  if ys[i] != ys[l] and sum(1 << j for j in range(p) if xs[i][j] != xs[l][j]) == e)]
                                  for e in clutter}
                        assert m == min(mass[u] for cs in covers.values() for u in cs)
                        count['deletion_radius_checks'] += 1
                        if not core:
                            continue
                        b = min(mass[u] for u in targets if targets[u][1] != core)
                        # Direct complete-multipartite formula for singleton graphs.
                        vals = []
                        for a in range(p):
                            if not core >> a & 1:
                                continue
                            groups = {}
                            for x, y, w in zip(xs, ys, ws):
                                key = tuple(x[j] for j in range(p) if j != a)
                                g = groups.setdefault(key, {})
                                g[y] = g.get(y, F(0)) + w
                            vals.append(sum((sum(g.values()) - max(g.values()) for g in groups.values()), F(0)))
                        assert b == min(vals)
                        count['core_formula_checks'] += 1
                        if m >= b:
                            continue
                        for K in [u for u in targets if mass[u] == m and targets[u][0] != fam]:
                            for Ka in [u for u in targets if mass[u] == b and targets[u][1] != core]:
                                count['separating_classes'] += 1
                                assert targets[K][1] == core
                                A = Ka & ~K
                                t = mass[A]
                                eta = min(m, b - m)
                                assert t >= b - m > 0
                                for deleted in [0, K, Ka]:
                                    weights = [F(0) if deleted >> i & 1 else w / (1 - mass[deleted]) for i, w in enumerate(ws)]
                                    for n in range(1, 5):
                                        err = F(0)
                                        for T, prob in observed_law(weights, n).items():
                                            out = targets[Ka][0] if not T & A else targets[K][0] if not T & K else fam
                                            if out != targets[deleted][0]:
                                                err += prob
                                        expected = ((1-t)**n + (1-m)**n - (1-t-m)**n if deleted == 0
                                                    else (1-t/(1-m))**n if deleted == K else F(0))
                                        assert err == expected, (xs, ys, ws, K, Ka, deleted, n, err, expected)
                                        assert err <= 2 * (1-eta)**n
                                        count['risk_checks'] += 1
    result = {'verdict': 'PASS', 'violations': 0, 'counts': count,
              'scope': 'All labelled supports in listed finite ranges; two mass vectors per support; n=1..4; all minimum changing deletions; post-hoc.'}
    Path(__file__).with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
