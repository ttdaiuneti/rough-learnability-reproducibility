"""Exact finite checks for the claims added or tightened in the mathematical revision.

Self-contained: structures are recomputed from literal sufficiency over attribute subsets, and
witness covers from discernibility sets. Nothing here is imported from the other checkers, and no
check is a substitute for the proofs; these are tests for the statements that changed.

C1  constant target: nu_Psi = +infinity  =>  the completed plug-in Psi(O_n u S0) never errs (n >= 1).
C2  finite radius: 0 < nu_Psi < 1 whenever nu_Psi < infinity.
C3  support deletion criterion (Proposition 13(a), support form): for T and U subset of T with
    T\\U nonempty, the family changes iff U contains a cover of G_e[T] for some e in the clutter of
    T, and the core changes iff U contains a cover of G_{a}[T] for some a in CORE_T.
C4  reduction (Theorem 17(a)): the minimum over triples equals the minimum over the reduced
    supports S0 u {x,x'}, computed by enumerating the at most four optional subsets.
C5  Corollary 18 constants, including the infinite-radius case.
C6  Proposition 12 sandwich, both inequalities, for the family and the core.
C7  Proposition 11 on a full Cartesian support.
C8  Corollary 15(a) converse construction with eps <= beta/(2(1+beta)).
"""
import itertools, json, math, random
from pathlib import Path

OUT = Path(__file__).with_name('check_revision_claims.json')
rng = random.Random(20260916)
TOL = 1e-10
stats = {k: 0 for k in ('c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8')}
violations = []


def bad(msg):
    violations.append(msg)


def subsets(it):
    it = list(it)
    for k in range(len(it) + 1):
        yield from itertools.combinations(it, k)


def family(T, lab, p):
    atoms = list(T)
    suff = []
    for B in subsets(range(p)):
        if all(lab[x] == lab[y] or any(x[b] != y[b] for b in B)
               for x, y in itertools.combinations(atoms, 2)):
            suff.append(frozenset(B))
    return frozenset(B for B in suff if not any(A < B for A in suff))


def core(fam):
    it = iter(fam)
    c = set(next(it))
    for R in it:
        c &= R
    return frozenset(c)


def struct(T, lab, p, target):
    f = family(tuple(T), lab, p)
    return f if target == 'fam' else core(f)


def D(x, y):
    return frozenset(i for i in range(len(x)) if x[i] != y[i])


def hsets(T, lab):
    return {D(x, y) for x, y in itertools.combinations(T, 2) if lab[x] != lab[y]}


def clutter(H):
    return {e for e in H if not any(f < e for f in H)}


def wit_edges(T, lab, e):
    return [(x, y) for x, y in itertools.combinations(T, 2)
            if lab[x] != lab[y] and D(x, y) == e]


def covers(V, E):
    Vs = set(V)
    return all(x in Vs or y in Vs for x, y in E)


def target_edges(T, lab, target):
    C = clutter(hsets(T, lab))
    return C if target == 'fam' else {e for e in C if len(e) == 1}


def nu_brute(S, lab, S0, w, target, p):
    """minimum floor weight of an optional Psi-changing deletion, by literal recomputation."""
    best, arg = math.inf, None
    optional = [x for x in S if x not in S0]
    for extra in subsets(optional):
        T = tuple(S0) + extra
        if not T:
            continue
        sT = struct(T, lab, p, target)
        for V in subsets(extra):
            if not V or len(V) == len(T):
                continue
            rest = tuple(x for x in T if x not in V)
            if struct(rest, lab, p, target) != sT:
                wv = sum(w[x] for x in V)
                if wv < best - 1e-15:
                    best, arg = wv, (T, V)
    return best, arg


def nu_reduced_algorithm(S, lab, S0, w, target):
    """Theorem 17(a) as an algorithm: one reduced support per oppositely labelled pair."""
    best = math.inf
    for x, y in itertools.combinations(S, 2):
        if lab[x] == lab[y]:
            continue
        e = D(x, y)
        if target == 'core' and len(e) != 1:
            continue
        T = tuple(dict.fromkeys(tuple(S0) + (x, y)))
        if e not in target_edges(T, lab, target):
            continue
        E = wit_edges(T, lab, e)
        optional = [z for z in (x, y) if z not in S0]
        for V in subsets(optional):                      # at most four subsets
            if V and covers(V, E):
                best = min(best, sum(w[z] for z in V))
    return best


def observed_law(Q, n):
    atoms = list(Q)
    out = {}
    for O in subsets(atoms):
        s = 0.0
        for J in subsets(O):
            s += (-1) ** (len(O) - len(J)) * (sum(Q[x] for x in J) ** n)
        out[frozenset(O)] = s
    return out


def plugin_error(Q, lab, S0, p, n, target):
    T = tuple(Q)
    truth = struct(T, lab, p, target)
    err = 0.0
    for O, pr in observed_law(Q, n).items():
        if pr < 1e-18:
            continue
        sup = tuple(dict.fromkeys(tuple(S0) + tuple(O)))
        est = struct(sup, lab, p, target) if sup else (
            frozenset([frozenset()]) if target == 'fam' else frozenset())
        if est != truth:
            err += pr
    return err


def random_floor(S):
    raw = [rng.random() + 0.05 for _ in S]
    scale = rng.uniform(0.3, 1.0) / sum(raw)
    return {x: r * scale for x, r in zip(S, raw)}


def members(S, lab, S0, w, how_many):
    out = []
    optional = [x for x in S if x not in S0]
    for extra in subsets(optional):
        T = tuple(S0) + extra
        if not T:
            continue
        left = 1 - sum(w[x] for x in T)
        Q = {x: w[x] for x in T}
        heavy = T[rng.randrange(len(T))]
        Q[heavy] += left
        out.append(Q)
    return rng.sample(out, min(len(out), how_many))


def labelled_supports(p, labels, sizes, cap=None):
    grid = list(itertools.product((0, 1), repeat=p))
    out = []
    for k in sizes:
        for S in itertools.combinations(grid, k):
            for labs in itertools.product(labels, repeat=k):
                if len(set(labs)) > 1:
                    out.append((S, dict(zip(S, labs))))
    if cap and len(out) > cap:
        out = rng.sample(out, cap)
    return out


# ---------------------------------------------------------------- C1-C5
def floor_class_checks(p, labels, sizes, n_list, cap=None, s0_draws=2):
    for S, lab in labelled_supports(p, labels, sizes, cap):
        for _ in range(s0_draws):
            S0 = () if rng.random() < 0.5 else tuple(rng.sample(list(S), rng.randint(1, len(S) - 1)))
            w = random_floor(S)
            for target in ('fam', 'core'):
                nu, arg = nu_brute(S, lab, S0, w, target, p)
                stats['c4'] += 1
                red = nu_reduced_algorithm(S, lab, S0, w, target)
                if not (abs(nu - red) < TOL or nu == red == math.inf):
                    bad(f'C4 reduction {target} S={S} S0={S0}: brute={nu} reduced={red}')
                if nu < math.inf:
                    stats['c2'] += 1
                    if not (0 < nu < 1 - 1e-12):
                        bad(f'C2 radius out of (0,1) {target} S={S} S0={S0}: {nu}')
                for Q in members(S, lab, S0, w, 3):
                    for n in n_list:
                        err = plugin_error(Q, lab, S0, p, n, target)
                        if nu == math.inf:
                            stats['c1'] += 1
                            if err > TOL:
                                bad(f'C1 constant target has error {target} S={S} S0={S0} n={n}: {err}')
                        else:
                            stats['c5'] += 1
                            if err > 2 ** len([x for x in S if x not in S0]) * (1 - nu) ** n + TOL:
                                bad(f'C5 upper bound {target} S={S} S0={S0} n={n}: {err}')
                            if not S0:
                                const = 2 * p if target == 'core' else 2 * math.comb(p, p // 2)
                                if err > const * (1 - nu) ** n + TOL:
                                    bad(f'C5 corollary constant {target} S={S} n={n}: {err}')


# ---------------------------------------------------------------- C3
def deletion_criterion_checks(p, labels, sizes, cap=None):
    for S, lab in labelled_supports(p, labels, sizes, cap):
        for target in ('fam', 'core'):
            sS = struct(S, lab, p, target)
            edges = target_edges(S, lab, target)
            for U in subsets(S):
                if not U or len(U) == len(S):
                    continue
                rest = tuple(x for x in S if x not in U)
                changed = struct(rest, lab, p, target) != sS
                criterion = any(covers(U, wit_edges(S, lab, e)) for e in edges)
                stats['c3'] += 1
                if changed != criterion:
                    bad(f'C3 deletion criterion {target} S={S} U={U}: changed={changed} criterion={criterion}')


# ---------------------------------------------------------------- C6
def plugin_sandwich_checks(p, labels, sizes, n_list, cap=None):
    for S, lab in labelled_supports(p, labels, sizes, cap):
        m = {x: rng.random() + 0.05 for x in S}
        z = sum(m.values())
        m = {x: v / z for x, v in m.items()}
        for target in ('fam', 'core'):
            edges = target_edges(S, lab, target)
            if not edges:
                continue
            terms, radius = [], math.inf
            for e in edges:
                E = wit_edges(S, lab, e)
                cvs = [V for V in subsets(S) if covers(V, E)]
                minimal = [V for V in cvs if not any(set(W) < set(V) for W in cvs)]
                for V in minimal:
                    terms.append(sum(m[x] for x in V))
                radius = min(radius, min(sum(m[x] for x in V) for V in minimal))
            for n in n_list:
                err = plugin_error(m, lab, (), p, n, target)
                upper = sum((1 - t) ** n for t in terms)
                stats['c6'] += 1
                if not ((1 - radius) ** n - TOL <= err <= upper + TOL):
                    bad(f'C6 sandwich {target} S={S} n={n}: {(1-radius)**n} <= {err} <= {upper}')


# ---------------------------------------------------------------- C7
def full_support_checks(p, labels):
    grid = list(itertools.product((0, 1), repeat=p))
    for labs in itertools.product(labels, repeat=len(grid)):
        if len(set(labs)) < 2:
            continue
        lab = dict(zip(grid, labs))
        ess = frozenset(i for i in range(p)
                        if any(lab[x] != lab[tuple(v ^ (1 if j == i else 0) for j, v in enumerate(x))]
                               for x in grid))
        fam = family(tuple(grid), lab, p)
        stats['c7'] += 1
        if fam != frozenset([ess]) or core(fam) != ess:
            bad(f'C7 full support labs={labs}: family={sorted(map(sorted, fam))} ess={sorted(ess)}')


# ---------------------------------------------------------------- C8
def corollary15_construction_checks(p, labels, sizes, cap=None):
    for S, lab in labelled_supports(p, labels, sizes, cap):
        C = clutter(hsets(S, lab))
        cor = [e for e in C if len(e) == 1]
        if not cor:
            continue
        for e in C:
            E = wit_edges(S, lab, e)
            for K in subsets(S):
                if not K or not covers(K, E) or len(K) == len(S):
                    continue
                preserving = all(any(x not in K and y not in K for x, y in wit_edges(S, lab, a))
                                 for a in cor)
                if not preserving:
                    continue
                outside = [x for x in S if x not in K]
                p0 = {x: 1 / len(outside) for x in outside}
                beta = min(p0.values())
                eps = beta / (2 * (1 + beta))
                mass = {x: (1 - eps) * p0[x] for x in outside}
                mass.update({x: eps / len(K) for x in K})

                def mincov(edge):
                    E2 = wit_edges(S, lab, edge)
                    return min(sum(mass[x] for x in V) for V in subsets(S) if covers(V, E2))
                mu_fam = min(mincov(f) for f in C)
                mu_core = min(mincov(a) for a in cor)
                stats['c8'] += 1
                if not (mu_fam <= eps + TOL and mu_fam < mu_core - TOL
                        and mu_core - mu_fam >= beta / 2 - TOL):
                    bad(f'C8 construction S={S} K={K}: mu_fam={mu_fam} mu_core={mu_core} beta={beta}')
                break


if __name__ == '__main__':
    floor_class_checks(3, (0, 1), (2, 3, 4), (1, 2, 4))
    floor_class_checks(3, (0, 1, 2), (2, 3, 4), (1, 3), cap=600)
    floor_class_checks(4, (0, 1), (2, 3, 4), (1, 3), cap=600)
    deletion_criterion_checks(3, (0, 1), (2, 3, 4))
    deletion_criterion_checks(3, (0, 1, 2), (3, 4), cap=400)
    deletion_criterion_checks(4, (0, 1), (3, 4), cap=400)
    plugin_sandwich_checks(3, (0, 1), (3, 4), (1, 2, 5), cap=300)
    plugin_sandwich_checks(4, (0, 1), (4,), (1, 3), cap=150)
    full_support_checks(2, (0, 1))
    full_support_checks(3, (0, 1))
    corollary15_construction_checks(3, (0, 1), (3, 4, 5), cap=300)
    corollary15_construction_checks(4, (0, 1), (4,), cap=200)
    out = dict(stats)
    out['n_violations'] = len(violations)
    out['violations'] = violations[:40]
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != 'violations'}, indent=1))
    for v in violations[:10]:
        print(v)
