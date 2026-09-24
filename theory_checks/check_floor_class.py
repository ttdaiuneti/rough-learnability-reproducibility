"""Independent exact checks for candidate Theorem 17 (structural-zero floor classes)
and a re-check of the Theorem 14 estimator errors.

Imports no project checker. Structures are computed by literal sufficiency over
attribute subsets; witness covers are computed separately from discernibility sets.
Writes only into this directory.
"""
import itertools, json, math, random, sys
from pathlib import Path

OUT = Path(__file__).with_name('check_floor_class.json')
rng = random.Random(20260915)
TOL = 1e-10


def subsets(it):
    it = list(it)
    for k in range(len(it) + 1):
        yield from itertools.combinations(it, k)


# ---------- literal structures ----------
def family(T, lab, p):
    atoms = list(T)
    suff = []
    for B in subsets(range(p)):
        ok = True
        for x, y in itertools.combinations(atoms, 2):
            if lab[x] != lab[y] and all(x[b] == y[b] for b in B):
                ok = False
                break
        if ok:
            suff.append(frozenset(B))
    mins = [B for B in suff if not any(A < B for A in suff)]
    return frozenset(mins)


def core(fam):
    it = iter(fam)
    c = set(next(it))
    for R in it:
        c &= R
    return frozenset(c)


# ---------- witness-cover machinery ----------
def D(x, y):
    return frozenset(i for i in range(len(x)) if x[i] != y[i])


def hsets(T, lab):
    return {D(x, y) for x, y in itertools.combinations(T, 2) if lab[x] != lab[y]}


def clutter(H):
    return {e for e in H if not any(f < e for f in H)}


def wit_edges(T, lab, e):
    return [(x, y) for x, y in itertools.combinations(T, 2) if lab[x] != lab[y] and D(x, y) == e]


def min_optional_cover(T, lab, e, S0, w):
    E = wit_edges(T, lab, e)
    opt = [x for x in T if x not in S0]
    best = math.inf
    for V in subsets(opt):
        Vs = set(V)
        if all(x in Vs or y in Vs for x, y in E):
            best = min(best, sum(w[x] for x in V))
    return best


def nu_def(S, lab, S0, w, target):
    opt = [x for x in S if x not in S0]
    best = math.inf
    for extra in subsets(opt):
        T = tuple(S0) + extra
        H = hsets(T, lab)
        C = clutter(H)
        edges = C if target == 'fam' else {e for e in C if len(e) == 1}
        for e in edges:
            best = min(best, min_optional_cover(T, lab, e, S0, w))
    return best


def nu_brute(S, lab, S0, w, target, p):
    """min w(V) over supports T and optional V with Psi(T minus V) != Psi(T)."""
    opt = [x for x in S if x not in S0]
    best, arg = math.inf, None
    for extra in subsets(opt):
        T = tuple(S0) + extra
        if not T:
            continue
        fT = family(T, lab, p)
        for V in subsets(extra):
            if not V or len(V) == len(T):
                continue
            T2 = tuple(x for x in T if x not in V)
            f2 = family(T2, lab, p)
            diff = (f2 != fT) if target == 'fam' else (core(f2) != core(fT))
            if diff:
                wv = sum(w[x] for x in V)
                if wv < best - 1e-15:
                    best, arg = wv, (T, V)
    return best, arg


def nu_reduced(S, lab, S0, w, target):
    best = math.inf
    for x, y in itertools.combinations(S, 2):
        if lab[x] == lab[y]:
            continue
        e = D(x, y)
        if target == 'core' and len(e) != 1:
            continue
        T = tuple(dict.fromkeys(tuple(S0) + (x, y)))
        if e not in clutter(hsets(T, lab)):
            continue
        E = wit_edges(T, lab, e)
        for V in subsets([z for z in (x, y) if z not in S0]):
            Vs = set(V)
            if V and all(a in Vs or b in Vs for a, b in E):
                best = min(best, sum(w[z] for z in V))
    return best


def nu_pairs(S, lab, w, target):
    best = math.inf
    for x, y in itertools.combinations(S, 2):
        if lab[x] != lab[y] and (target == 'fam' or len(D(x, y)) == 1):
            best = min(best, min(w[x], w[y]))
    return best


# ---------- exact risks ----------
def prob_observed_exactly(Q, n):
    atoms = list(Q)
    out = {}
    for O in subsets(atoms):
        s = 0.0
        for J in subsets(O):
            s += (-1) ** (len(O) - len(J)) * (sum(Q[x] for x in J) ** n)
        out[frozenset(O)] = s
    return out


def estimator_error(Q, lab, S0, p, n, target):
    T = tuple(Q)
    truth = family(T, lab, p)
    truth = truth if target == 'fam' else core(truth)
    err = 0.0
    for O, pr in prob_observed_exactly(Q, n).items():
        if pr < 1e-18:
            continue
        sup = tuple(dict.fromkeys(tuple(S0) + tuple(O)))
        if not sup:
            est = frozenset([frozenset()]) if target == 'fam' else frozenset()
        else:
            f = family(sup, lab, p)
            est = f if target == 'fam' else core(f)
        if est != truth:
            err += pr
    return err


def overlap(Q1, Q2, n):
    atoms = sorted(set(Q1) | set(Q2))
    s = 0.0
    for seq in itertools.product(atoms, repeat=n):
        a = math.prod(Q1.get(x, 0.0) for x in seq)
        b = math.prod(Q2.get(x, 0.0) for x in seq)
        s += min(a, b)
    return s


def random_floor(S):
    raw = [rng.random() + 0.05 for _ in S]
    scale = rng.uniform(0.3, 1.0) / sum(raw)
    return {x: r * scale for x, r in zip(S, raw)}


def masses_on(T, w, heavy=None):
    """mass >= floor on T, leftover given to `heavy` (or spread)."""
    left = 1 - sum(w[x] for x in T)
    Q = {x: w[x] for x in T}
    if heavy is None:
        for x in T:
            Q[x] += left / len(T)
    else:
        Q[heavy] += left
    return Q


stats = dict(instances=0, nu_equal_checks=0, const_checks=0, order_checks=0,
             upper_checks=0, lower_constructions=0, pair_formula_checks=0,
             sharp_const_checks=0, thm14_checks=0, violations=[])


def bad(msg):
    stats['violations'].append(msg)


def run_domain(p, labels, max_support, n_list, per_instance_S0=2, sample_supports=None):
    grid = list(itertools.product((0, 1), repeat=p))
    supports = []
    for k in range(2, max_support + 1):
        for S in itertools.combinations(grid, k):
            for labs in itertools.product(labels, repeat=k):
                if len(set(labs)) < 2:
                    continue
                supports.append((S, dict(zip(S, labs))))
    if sample_supports and len(supports) > sample_supports:
        supports = rng.sample(supports, sample_supports)
    for S, lab in supports:
        S0_choices = [()]
        for _ in range(per_instance_S0 - 1):
            k0 = rng.randint(1, len(S) - 1)
            S0_choices.append(tuple(rng.sample(list(S), k0)))
        for S0 in S0_choices:
            w = random_floor(S)
            stats['instances'] += 1
            nus = {}
            for tg in ('fam', 'core'):
                a = nu_def(S, lab, S0, w, tg)
                b, arg = nu_brute(S, lab, S0, w, tg, p)
                c = nu_reduced(S, lab, S0, w, tg)
                stats['nu_equal_checks'] += 1
                if not (abs(a - b) < TOL or a == b == math.inf) or not (abs(a - c) < TOL or a == c == math.inf):
                    bad(f'nu mismatch {tg} S={S} S0={S0}: def={a} brute={b} red={c}')
                if not S0:
                    d = nu_pairs(S, lab, w, tg)
                    stats['pair_formula_checks'] += 1
                    if not (abs(a - d) < TOL or a == d == math.inf):
                        bad(f'pair formula {tg} S={S}: {a} vs {d}')
                # (i) constancy
                opt = [x for x in S if x not in S0]
                vals = set()
                for extra in subsets(opt):
                    T = tuple(S0) + extra
                    if not T:
                        continue
                    f = family(T, lab, p)
                    vals.add(f if tg == 'fam' else core(f))
                stats['const_checks'] += 1
                if (len(vals) == 1) != (a == math.inf):
                    bad(f'constancy {tg} S={S} S0={S0} nu={a} nvals={len(vals)}')
                nus[tg] = (a, arg)
            stats['order_checks'] += 1
            if nus['fam'][0] > nus['core'][0] + TOL:
                bad(f'nu_fam > nu_core S={S} S0={S0}')
            # (ii) upper bounds on members, and lower-bound construction
            opt = [x for x in S if x not in S0]
            members = []
            for extra in subsets(opt):
                T = tuple(S0) + extra
                if len(T) >= 1:
                    members.append(masses_on(T, w))
                    members.append(masses_on(T, w, heavy=T[rng.randrange(len(T))]))
            members = rng.sample(members, min(len(members), 6))
            for tg in ('fam', 'core'):
                nu, arg = nus[tg]
                if nu == math.inf:
                    continue
                N = 2 ** len(opt)
                for Q in members:
                    for n in n_list:
                        err = estimator_error(Q, lab, S0, p, n, tg)
                        stats['upper_checks'] += 1
                        if err > N * (1 - nu) ** n + TOL:
                            bad(f'upper {tg} S={S} S0={S0} n={n} err={err} bound={N*(1-nu)**n}')
                        if not S0:
                            M = 2 * p if tg == 'core' else 2 * math.comb(p, p // 2)
                            stats['sharp_const_checks'] += 1
                            if err > M * (1 - nu) ** n + TOL:
                                bad(f'sharp upper {tg} S={S} n={n} err={err}')
                T, V = arg
                Q1 = masses_on(T, w, heavy=[x for x in T if x not in V][0])
                for x in V:
                    Q1[x] = w[x]
                rest = [x for x in T if x not in V]
                left = 1 - sum(Q1.values())
                Q1[rest[0]] += left
                Z = sum(Q1[x] for x in rest)
                Q2 = {x: Q1[x] / Z for x in rest}
                ok = all(Q1[x] >= w[x] - TOL for x in T) and all(Q2[x] >= w[x] - TOL for x in rest)
                ok &= set(S0) <= set(rest)
                f1, f2 = family(T, lab, p), family(tuple(rest), lab, p)
                ok &= (f1 != f2) if tg == 'fam' else (core(f1) != core(f2))
                for n in (1, 2, 3):
                    if len(T) ** n <= 5000:
                        ok &= abs(overlap(Q1, Q2, n) - (1 - nu) ** n) < 1e-9
                stats['lower_constructions'] += 1
                if not ok:
                    bad(f'lower construction {tg} S={S} S0={S0}')


def thm14_checks(p, labels, max_support, count):
    grid = list(itertools.product((0, 1), repeat=p))
    done = 0
    tries = 0
    while done < count and tries < 200000:
        tries += 1
        k = rng.randint(3, max_support)
        S = tuple(rng.sample(grid, k))
        lab = {x: rng.choice(labels) for x in S}
        if len(set(lab.values())) < 2:
            continue
        m = {x: rng.random() + 0.01 for x in S}
        z = sum(m.values()); m = {x: v / z for x, v in m.items()}
        H = hsets(S, lab); C = clutter(H)
        cor = [e for e in C if len(e) == 1]
        if not cor:
            continue
        def mincov(e):
            E = wit_edges(S, lab, e); best = (math.inf, None)
            for V in subsets(S):
                Vs = set(V)
                if all(a in Vs or b in Vs for a, b in E):
                    wv = sum(m[x] for x in V)
                    if wv < best[0]:
                        best = (wv, V)
            return best
        mf, K = min((mincov(e) for e in C), key=lambda t: t[0])
        mc, Ka = min((mincov(e) for e in cor), key=lambda t: t[0])
        if not mf < mc - 1e-9:
            continue
        A = [x for x in Ka if x not in K]
        def cond(U):
            keep = [x for x in S if x not in U]; zz = sum(m[x] for x in keep)
            return {x: m[x] / zz for x in keep}
        P, Qd, R = dict(m), cond(K), cond(Ka)
        fams = {id_: family(tuple(D_), lab, p) for id_, D_ in (('P', P), ('Q', Qd), ('R', R))}
        t = sum(m[x] for x in A)
        eta = min(mf, mc - mf)
        for n in (1, 2, 3, 4):
            for name, law in (('P', P), ('Q', Qd), ('R', R)):
                err = 0.0
                for O, pr in prob_observed_exactly(law, n).items():
                    if not any(x in O for x in A):
                        est = fams['R']
                    elif not any(x in O for x in K):
                        est = fams['Q']
                    else:
                        est = fams['P']
                    if est != fams[name]:
                        err += pr
                formula = {'P': (1 - t) ** n + (1 - mf) ** n - (1 - t - mf) ** n,
                           'Q': (1 - t / (1 - mf)) ** n, 'R': 0.0}[name]
                stats['thm14_checks'] += 1
                if abs(err - formula) > 1e-9 or err > 2 * (1 - eta) ** n + 1e-12:
                    bad(f'thm14 {name} n={n} S={S} err={err} formula={formula}')
        done += 1


if __name__ == '__main__':
    run_domain(3, (0, 1), 4, (1, 2, 4, 8), per_instance_S0=2)
    run_domain(3, (0, 1, 2), 4, (1, 3, 6), per_instance_S0=2, sample_supports=1500)
    run_domain(4, (0, 1), 4, (1, 3, 6), per_instance_S0=2, sample_supports=1500)
    run_domain(3, (0, 1), 5, (2, 5), per_instance_S0=2, sample_supports=800)
    thm14_checks(3, (0, 1), 5, 150)
    thm14_checks(3, (0, 1, 2), 5, 100)
    thm14_checks(4, (0, 1), 5, 100)
    stats['n_violations'] = len(stats['violations'])
    stats['violations'] = stats['violations'][:50]
    OUT.write_text(json.dumps(stats, indent=1))
    print(json.dumps({k: v for k, v in stats.items() if k != 'violations'}, indent=1))
    print('\n'.join(stats['violations'][:10]))
