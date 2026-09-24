"""Numbers of the worked examples in Section 4 (Examples 1-4), computed exactly.

Structures come from the discernibility sets of the labelled support; witness-graph covers are
enumerated; probabilities are exact rationals, converted to floats only for printing and for
the logarithms in the sample-size bounds. Writes worked_examples.json next to this file.
"""
import itertools, json, math, os
from fractions import Fraction as F

DELTA = F(1, 100)
L = math.log(1 / (2 * DELTA))


def disc(x, y):
    return frozenset(a for a in range(len(x)) if x[a] != y[a])


def structure(S, f):
    H = {disc(x, y) for x, y in itertools.combinations(S, 2) if f[x] != f[y]}
    C = sorted([e for e in H if not any(g < e for g in H)], key=lambda e: (len(e), sorted(e)))
    core = sorted(next(iter(e)) for e in C if len(e) == 1)
    return H, C, core


def wgraph(S, f, e):
    return [(x, y) for x, y in itertools.combinations(sorted(S), 2) if f[x] != f[y] and disc(x, y) == e]


def minimal_covers(S, E, allowed=None):
    pool = sorted(S) if allowed is None else sorted(allowed)
    out = []
    for r in range(1, len(pool) + 1):
        for K in itertools.combinations(pool, r):
            if all(x in K or y in K for x, y in E) and not any(set(B) <= set(K) for B in out):
                out.append(K)
    return out


def prob_union(events, mass, n):
    """P(at least one K in events has K inside the unobserved set), inclusion-exclusion."""
    tot = F(0)
    for r in range(1, len(events) + 1):
        for sub in itertools.combinations(events, r):
            U = set().union(*map(set, sub))
            tot += (-1) ** (r + 1) * (1 - sum(mass[x] for x in U)) ** n
    return tot


def ceil_log(num, rate):
    return math.ceil(math.log(num) / -math.log(1 - rate))


out = {}

# ---------------- Example 1: exact plug-in sample sizes on P_eps (Section 4.1), delta = 0.01
eps = F(1, 1000)
x0, x1, x2, x3 = 'x0', 'x1', 'x2', 'x3'
m = {x0: F(1, 4), x1: F(1, 2) - eps, x2: F(1, 4), x3: eps}
def core_err(n):   # sup over the class of the plug-in core error (Proposition 3)
    e01 = 2 * F(3, 4) ** n - F(1, 2) ** n
    e2 = F(3, 4) ** n + (F(1, 2) + eps) ** n - (F(1, 4) + eps) ** n
    return max(e01, e2)
def fam_err(n):    # P0 needs x0, x2, x3; P1 needs x0, x2; P2 needs x0, x1
    p0 = prob_union([(x0,), (x2,), (x3,)], m, n)
    return max(p0, core_err(n))
n = 1
while core_err(n) > DELTA: n += 1
n_core_plugin = n
lo, hi = 1, 20000
while lo < hi:
    mid = (lo + hi) // 2
    if fam_err(mid) <= DELTA: hi = mid
    else: lo = mid + 1
out['example1'] = dict(eps=str(eps), delta=str(DELTA), n_core_plugin=n_core_plugin,
                       n_core_UB=math.ceil(math.log(2 / DELTA) / math.log(4 / 3)),
                       n_core_LB=math.ceil(math.log(1 / (2 * DELTA)) / math.log(4)),
                       n_family_plugin=lo,
                       n_family_LB=math.ceil(L / -math.log(1 - eps)))

# ---------------- Examples 2 and 3: a population whose covers are not single atoms
S = [(0, 0, 0), (0, 1, 1), (1, 0, 0), (1, 1, 1)]
f = {(0, 0, 0): 0, (0, 1, 1): 1, (1, 0, 0): 1, (1, 1, 1): 0}
w = {(0, 0, 0): F(40, 100), (0, 1, 1): F(10, 100), (1, 0, 0): F(45, 100), (1, 1, 1): F(5, 100)}
H, C, core = structure(S, f)
name = lambda x: ''.join(map(str, x))
ex2 = dict(clutter=[sorted(e) for e in C], core=core)
radius = {}
covers = {}
for e in C:
    E = wgraph(S, f, e)
    Ks = minimal_covers(S, E)
    covers[e] = Ks
    radius[e] = min(sum(w[x] for x in K) for K in Ks)
    ex2['witness_graph_' + ''.join(str(a + 1) for a in sorted(e))] = [[name(x), name(y)] for x, y in E]
    ex2['min_covers_' + ''.join(str(a + 1) for a in sorted(e))] = {''.join('+' + name(x) for x in K)[1:]: str(sum(w[x] for x in K)) for K in Ks}
mu_fam = min(radius.values())
mu_core = min(radius[frozenset([a])] for a in core)
# Bayes risk tau_a of predicting Y without attribute a (should equal mu({a}))
tau = {}
for a in core:
    groups = {}
    for x in S:
        z = tuple(v for i, v in enumerate(x) if i != a)
        groups.setdefault(z, {}).setdefault(f[x], F(0))
        groups[z][f[x]] += w[x]
    tau[a] = sum(sum(g.values()) - max(g.values()) for g in groups.values())
ex2.update(mu_fam=str(mu_fam), mu_core=str(mu_core), tau={str(a + 1): str(v) for a, v in tau.items()},
           rarest_atom=str(min(w.values())))
fam_events = [K for e in C for K in covers[e]]
core_events = [K for a in core for K in covers[frozenset([a])]]
rows = []
for n in (10, 20, 50):
    rows.append(dict(n=n,
                     family_lower=float((1 - mu_fam) ** n),
                     family_exact=float(prob_union(fam_events, w, n)),
                     family_upper=float(sum((1 - sum(w[x] for x in K)) ** n for K in fam_events)),
                     core_lower=float((1 - mu_core) ** n),
                     core_exact=float(prob_union(core_events, w, n)),
                     core_upper=float(sum((1 - sum(w[x] for x in K)) ** n for K in core_events))))
ex2['prop12'] = rows
nf = 1
while prob_union(fam_events, w, nf) > DELTA: nf += 1
nc = 1
while prob_union(core_events, w, nc) > DELTA: nc += 1
ex2['plugin_n_at_delta'] = dict(family=nf, core=nc)
out['example2'] = ex2

# curves for the figure of Example 2: exact plug-in errors and the bounds of Proposition 12
import csv
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'example2_curves.csv'), 'w', newline='') as fh:
    wr = csv.writer(fh)
    wr.writerow(['n', 'family_lower', 'family_exact', 'family_upper', 'core_lower', 'core_exact', 'core_upper'])
    for n in range(1, 61):
        wr.writerow([n, float((1 - mu_fam) ** n), float(prob_union(fam_events, w, n)),
                     float(sum((1 - sum(w[x] for x in K)) ** n for K in fam_events)),
                     float((1 - mu_core) ** n), float(prob_union(core_events, w, n)),
                     float(sum((1 - sum(w[x] for x in K)) ** n for K in core_events))])

# Example 3: Theorem 14 on the same population
e_big = [e for e in C if len(e) > 1][0]
K = min(covers[e_big], key=lambda K: sum(w[x] for x in K))
a = core[0]
Ka = min(covers[frozenset([a])], key=lambda K: sum(w[x] for x in K))
core_preserving = all(any(x not in K and y not in K for x, y in wgraph(S, f, frozenset([b]))) for b in core)
def cond(keep):
    tot = sum(w[x] for x in keep)
    return [x for x in keep], {x: w[x] / tot for x in keep}
QK, _ = cond([x for x in S if x not in K])
QKa, _ = cond([x for x in S if x not in Ka])
eta = min(mu_fam, mu_core - mu_fam)
out['example3'] = dict(
    K=[name(x) for x in K], K_mass=str(sum(w[x] for x in K)), core_preserving=core_preserving,
    Ka=[name(x) for x in Ka], Ka_mass=str(sum(w[x] for x in Ka)),
    Q_K=dict(support=[name(x) for x in QK], clutter=[sorted(e) for e in structure(QK, f)[1]], core=structure(QK, f)[2]),
    Q_Ka=dict(support=[name(x) for x in QKa], clutter=[sorted(e) for e in structure(QKa, f)[1]], core=structure(QKa, f)[2]),
    eta=str(eta),
    family=[L / -math.log(1 - mu_fam), ceil_log(2 / DELTA, eta)],
    core=[L / -math.log(1 - mu_core), ceil_log(1 / DELTA, mu_core - mu_fam)])

# ---------------- Example 4: known configurations in a structural-zero class (Theorem 17)
def floor_radii(S, f, S0, w):
    nu = {'fam': None, 'core': None}
    opt = [x for x in S if x not in S0]
    for r in range(0, len(opt) + 1):
        for extra in itertools.combinations(opt, r):
            T = list(S0) + list(extra)
            if not T: continue
            _, CT, coreT = structure(T, f)
            for kind, edges in (('fam', CT), ('core', [frozenset([b]) for b in coreT])):
                for e in edges:
                    E = wgraph(T, f, e)
                    for V in minimal_covers(T, E, allowed=[x for x in T if x not in S0]):
                        val = sum(w[x] for x in V)
                        if nu[kind] is None or val < nu[kind]: nu[kind] = val
    return nu

p = 3
K_const = 2 * p          # = 2 * binom(3, 1): the same for core and family when p = 3
Sx = {'x0': (0, 0, 0), 'x1': (0, 0, 1), 'x2': (1, 0, 0), 'x3': (1, 1, 1)}
fx = {Sx['x0']: 0, Sx['x1']: 0, Sx['x2']: 1, Sx['x3']: 0}
eps4 = F(2, 100)
wx = {Sx['x0']: F(1, 4), Sx['x1']: F(1, 4), Sx['x2']: F(1, 4), Sx['x3']: eps4}
ex4 = {}
for label, S0 in (('none', []), ('x3', [Sx['x3']])):
    nu = floor_radii(list(Sx.values()), fx, S0, wx)
    ex4[label] = {k: dict(nu=str(v), n_lower=math.ceil(L / -math.log(1 - v)),
                          n_upper=ceil_log(K_const / DELTA, v)) for k, v in nu.items()}
out['example4'] = dict(eps=str(eps4), delta=str(DELTA), K=K_const, **ex4)

print(json.dumps(out, indent=1, default=str))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'worked_examples.json'), 'w'),
          indent=1, default=str)
