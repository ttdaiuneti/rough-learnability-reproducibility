"""Exact check of the constant in Theorem 17(b): for every Q in a structural-zero class,
P_Q(Psi_hat_n != Psi(Q)) <= 2|C_T| (1 - nu_fam)^n (family) and <= 2|CORE_T| (1 - nu_core)^n (core),
with Psi_hat_n = Psi(O_n u S0). Structures by literal sufficiency; probabilities exact (Fraction)."""
import itertools, random
from fractions import Fraction as F

def structures(T, f, p):
    T = list(T)
    H = set()
    for x, y in itertools.combinations(T, 2):
        if f[x] != f[y]:
            H.add(frozenset(a for a in range(p) if x[a] != y[a]))
    C = [e for e in H if not any(g < e for g in H)]
    suff = [frozenset(B) for r in range(p + 1) for B in itertools.combinations(range(p), r)
            if all(B_ & e for e in H for B_ in [frozenset(B)])]
    R = frozenset(B for B in suff if not any(B2 < B for B2 in suff))
    core = frozenset.intersection(*R) if R else frozenset()
    return H, C, R, core

def witness_edges(T, f, p, e):
    return [(x, y) for x, y in itertools.combinations(sorted(T), 2)
            if f[x] != f[y] and frozenset(a for a in range(p) if x[a] != y[a]) == e]

def is_cover(V, edges):
    return all(x in V or y in V for x, y in edges)

def radii(S, f, S0, w, p):
    nu = {'fam': None, 'core': None}
    opt = [x for x in S if x not in S0]
    for r in range(0, len(S) - len(S0) + 1):
        for extra in itertools.combinations(opt, r):
            T = set(S0) | set(extra)
            if not T: continue
            H, C, R, core = structures(T, f, p)
            cand = {'fam': C, 'core': [frozenset([a]) for a in core]}
            for k, edges_list in cand.items():
                for e in edges_list:
                    E = witness_edges(T, f, p, e)
                    Topt = [x for x in T if x not in S0]
                    for s in range(1, len(Topt) + 1):
                        for V in itertools.combinations(Topt, s):
                            if is_cover(set(V), E):
                                m = sum(w[x] for x in V)
                                if nu[k] is None or m < nu[k]: nu[k] = m
    return nu

def exact_error(Q, f, S0, p, n, target):
    T = sorted(Q)
    true = structures(set(T), f, p)
    truth = true[2] if target == 'fam' else true[3]
    err = F(0)
    for seq in itertools.product(T, repeat=n):
        O = set(seq) | set(S0)
        s = structures(O, f, p)
        est = s[2] if target == 'fam' else s[3]
        if est != truth:
            pr = F(1)
            for x in seq: pr *= Q[x]
            err += pr
    return err, true

rng = random.Random(7)
p = 3
cube = list(itertools.product((0, 1), repeat=p))
viol = checks = 0
for trial in range(250):
    S = rng.sample(cube, rng.randint(2, 4))
    f = {x: rng.randint(0, 1) for x in S}
    S0 = [x for x in S if rng.random() < 0.3]
    w = {x: F(rng.randint(1, 4), 4 * len(S) + rng.randint(0, 4)) for x in S}
    if sum(w.values()) > 1: continue
    nu = radii(S, f, S0, w, p)
    opt = [x for x in S if x not in S0]
    for r in range(0, len(opt) + 1):
        for extra in itertools.combinations(opt, r):
            T = list(S0) + list(extra)
            if not T: continue
            rest = 1 - sum(w[x] for x in T)
            if rest < 0: continue
            for star in T:
                Q = {x: w[x] + (rest if x == star else 0) for x in T}
                for n in range(1, 4):
                    for target in ('fam', 'core'):
                        if nu[target] is None: continue
                        err, (H, C, R, core) = exact_error(Q, f, S0, p, n, target)
                        K = len(C) if target == 'fam' else len(core)
                        bound = 2 * K * (1 - nu[target]) ** n
                        checks += 1
                        if err > bound: viol += 1; print('VIOLATION', S, f, S0, T, n, target, err, bound)
print('checks', checks, 'violations', viol)
import json, os
json.dump(dict(scope='three binary attributes, binary labels, random supports of 2-4 atoms, random known sets S0 and floors (seed 7), every admissible support with each atom in turn carrying the residual mass, n = 1..3, both targets',
               checks=checks, violations=viol, verdict='PASS' if viol == 0 else 'FAIL'),
          open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'check_thm17_constant.json'), 'w'), indent=1)
