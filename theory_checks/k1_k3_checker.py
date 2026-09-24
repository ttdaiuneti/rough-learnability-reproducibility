"""Pre-registered checks K1-K3 of theory_spec.md (SHA-256 in ../spec_hash.txt).

Everything in exact rationals. Reduct families are recomputed LITERALLY from the definition of
sufficiency on the observed atoms, never through the realisation criterion being tested.

Exact plug-in error: with g(U) = P(T_n subset of U) = p(U)^n,
    P(error) = sum_T ind(T) P(T_n = T) = sum_U c(U) p(U)^n,
    c(U) = sum_{T superset of U} (-1)^{|T|-|U|} ind(T)       (Mobius inversion; c is integer and
                                                           depends only on the labeled support).
The Mobius route is cross-checked against brute-force enumeration of all |S|^n sequences.
"""
import argparse
import itertools
import json
import os
import random
import sys
import time
from fractions import Fraction as F

HERE = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------------------- rough structures
def disc(u, v):
    return frozenset(i for i, (a, b) in enumerate(zip(u, v)) if a != b)


def family_literal(atoms, lab, p):
    suff = []
    for r in range(p + 1):
        for B in itertools.combinations(range(p), r):
            seen = {}
            ok = True
            for x in atoms:
                key = tuple(x[i] for i in B)
                if key in seen and seen[key] != lab[x]:
                    ok = False
                    break
                seen.setdefault(key, lab[x])
            if ok:
                suff.append(frozenset(B))
    return frozenset(B for B in suff if not any(D < B for D in suff))


def core_of(fam):
    return frozenset.intersection(*fam) if fam else frozenset()


def clutter_of(atoms, lab):
    E = {disc(u, v) for u, v in itertools.combinations(atoms, 2) if lab[u] != lab[v]}
    return frozenset(e for e in E if not any(d < e for d in E))


# ----------------------------------------------------------------------------- subsets as masks
def members(mask, k):
    return [i for i in range(k) if mask >> i & 1]


def popcount(m):
    return bin(m).count("1")


def mass_of(mask, masses):
    return sum((masses[i] for i in members(mask, len(masses))), F(0))


def mobius_coeffs(ind, k):
    """c(U) = sum_{T >= U} (-1)^{|T|-|U|} ind(T), for all masks U of a k-set."""
    full = (1 << k) - 1
    c = [0] * (1 << k)
    for U in range(1 << k):
        rest = full & ~U
        s = 0
        sub = rest
        while True:
            T = U | sub
            if ind[T]:
                s += -1 if popcount(sub) % 2 else 1
            if sub == 0:
                break
            sub = (sub - 1) & rest
        c[U] = s
    return c


def exact_error(c, masses, n):
    k = len(masses)
    tot = F(0)
    for U in range(1 << k):
        if c[U]:
            tot += c[U] * mass_of(U, masses) ** n
    return tot


def brute_error(ind, masses, n):
    k = len(masses)
    tot = F(0)
    for seq in itertools.product(range(k), repeat=n):
        T = 0
        pr = F(1)
        for i in seq:
            T |= 1 << i
            pr *= masses[i]
        if ind[T]:
            tot += pr
    return tot


# ----------------------------------------------------------------------------- covers
def witness_pairs(atoms, lab, e):
    k = len(atoms)
    return [(i, j) for i, j in itertools.combinations(range(k), 2)
            if lab[atoms[i]] != lab[atoms[j]] and disc(atoms[i], atoms[j]) == e]


def is_cover(mask, pairs):
    return all((mask >> i & 1) or (mask >> j & 1) for i, j in pairs)


def minimal_covers(pairs, k):
    covers = [m for m in range(1 << k) if is_cover(m, pairs)]
    cs = set(covers)
    return [m for m in covers if not any((m & ~(1 << i)) in cs for i in members(m, k))]


# ----------------------------------------------------------------------------- one support
class Support:
    def __init__(self, atoms, lab, p):
        self.atoms, self.lab, self.p, self.k = list(atoms), dict(lab), p, len(atoms)
        self.fam = family_literal(self.atoms, self.lab, p)
        self.core = core_of(self.fam)
        self.clutter = clutter_of(self.atoms, self.lab)
        self.fam_T = {}
        for T in range(1, 1 << self.k):
            sub = [self.atoms[i] for i in members(T, self.k)]
            f = family_literal(sub, self.lab, p)
            self.fam_T[T] = (f, core_of(f))
        ind_f = [False] + [self.fam_T[T][0] != self.fam for T in range(1, 1 << self.k)]
        ind_c = [False] + [self.fam_T[T][1] != self.core for T in range(1, 1 << self.k)]
        self.c_fam = mobius_coeffs(ind_f, self.k)
        self.c_core = mobius_coeffs(ind_c, self.k)
        self.ind_f, self.ind_c = ind_f, ind_c
        self.pairs = {e: witness_pairs(self.atoms, self.lab, e) for e in self.clutter}
        self.mincov = {e: minimal_covers(self.pairs[e], self.k) for e in self.clutter}

    def restrict(self, removed_mask):
        keep = [self.atoms[i] for i in range(self.k) if not removed_mask >> i & 1]
        return keep


def random_masses(rng, k, kind):
    if kind == "rare" and k >= 2:
        w = [rng.randint(30, 60) for _ in range(k)]
        w[rng.randrange(k)] = 1
    else:
        w = [rng.randint(1, 20) for _ in range(k)]
    s = sum(w)
    return [F(x, s) for x in w]


def supports(p, labels, kmin, kmax):
    atoms = list(itertools.product((0, 1), repeat=p))
    for k in range(kmin, kmax + 1):
        for S in itertools.combinations(atoms, k):
            for ys in itertools.product(labels, repeat=k):
                if len(set(ys)) >= 2:
                    yield S, dict(zip(S, ys))


# ----------------------------------------------------------------------------- main checks
def run_config(name, p, labels, kmin, kmax, mass_kinds, nmax, rng, out, brute_budget):
    stats = dict(config=name, p=p, labels=list(labels), support_sizes=[kmin, kmax], nmax=nmax,
                 supports=0, mass_vectors=0, k1_cases=0, k1_violations=0,
                 min_err_over_lower_fam=None, max_err_over_upper_fam=None,
                 min_err_over_lower_core=None, max_err_over_upper_core=None,
                 lemma1_iii_violations=0, ordering_violations=0, empty_core_nonzero_error=0,
                 k2_checks=0, k2_violations=0, k2_twopoint_checks=0, k2_twopoint_violations=0,
                 k3_supports_with_core=0, k3_supports_admitting_cp_cover=0,
                 k3_structure_checks=0, k3_checks=0, k3_violations=0,
                 brute_crosschecks=0, brute_mismatches=0)
    viol = out.setdefault("violations", [])

    def record(kind, detail):
        if len(viol) < 50:
            viol.append(dict(config=name, kind=kind, **detail))

    def upd(key, val, mode):
        cur = stats[key]
        if cur is None or (mode == "min" and val < cur) or (mode == "max" and val > cur):
            stats[key] = val

    for S, lab in supports(p, labels, kmin, kmax):
        sup = Support(S, lab, p)
        stats["supports"] += 1
        core_from_clutter = frozenset(next(iter(e)) for e in sup.clutter if len(e) == 1)
        if core_from_clutter != sup.core:
            stats["lemma1_iii_violations"] += 1
            record("lemma1_iii", dict(S=S, lab=list(lab.values())))
        core_edges = [e for e in sup.clutter if len(e) == 1]

        # structural K2 (mass-free part) and K3 structure
        qcache = {}

        def sub_support(removed):
            if removed not in qcache:
                keep = sup.restrict(removed)
                qcache[removed] = Support(keep, {x: lab[x] for x in keep}, p) if keep else None
            return qcache[removed]

        for e in sup.clutter:
            for K in sup.mincov[e]:
                Q = sub_support(K)
                stats["k2_checks"] += 1
                bad = Q is None or Q.fam == sup.fam or (len(e) == 1 and next(iter(e)) in Q.core)
                if bad:
                    stats["k2_violations"] += 1
                    record("k2_structure", dict(S=S, lab=list(lab.values()), e=sorted(e), K=K))

        if sup.core:
            stats["k3_supports_with_core"] += 1
            admits = False
            for e in sup.clutter:
                for K in sup.mincov[e]:
                    cp = all(any(not (K >> i & 1) and not (K >> j & 1) for i, j in sup.pairs[frozenset([b])])
                             for b in sup.core)
                    if not cp:
                        continue
                    admits = True
                    Q = sub_support(K)
                    for a in sup.core:
                        ea = frozenset([a])
                        for Ka in sup.mincov[ea]:
                            R = sub_support(Ka)
                            stats["k3_structure_checks"] += 1
                            problems = []
                            if Q is None or Q.core != sup.core:
                                problems.append("CORE_Q != CORE_P")
                            if Q is None or Q.fam == sup.fam:
                                problems.append("R_Q == R_P")
                            if len(e) < 2:
                                problems.append("e singleton")
                            if (Ka & ~K) == 0:
                                problems.append("K_a subset of K")
                            if R is None or a in R.core or R.core == sup.core:
                                problems.append("a in CORE_R or CORE_R == CORE_P")
                            if problems:
                                stats["k3_violations"] += 1
                                record("k3_structure", dict(S=S, lab=list(lab.values()), e=sorted(e), K=K, a=a, Ka=Ka, problems=problems))
                                continue
            if admits:
                stats["k3_supports_admitting_cp_cover"] += 1

        kinds = mass_kinds
        for kind in kinds:
            masses = random_masses(rng, sup.k, kind)
            stats["mass_vectors"] += 1
            mu = {e: min(mass_of(K, masses) for K in sup.mincov[e]) for e in sup.clutter}
            mu_fam = min(mu.values()) if mu else None
            mu_core = min((mu[e] for e in core_edges), default=None)
            if mu_fam is not None and mu_core is not None and mu_fam > mu_core:
                stats["ordering_violations"] += 1
                record("ordering", dict(S=S, masses=[str(m) for m in masses]))
            for n in range(1, nmax + 1):
                err_f = exact_error(sup.c_fam, masses, n)
                err_c = exact_error(sup.c_core, masses, n)
                stats["k1_cases"] += 1
                # family
                low = (1 - mu_fam) ** n
                mid = sum(((1 - mass_of(K, masses)) ** n for e in sup.clutter for K in sup.mincov[e]), F(0))
                up = sum((len(sup.mincov[e]) * (1 - mu[e]) ** n for e in sup.clutter), F(0))
                ok_f = low <= err_f <= mid <= up
                upd("min_err_over_lower_fam", float(err_f / low), "min")
                upd("max_err_over_upper_fam", float(err_f / mid) if mid else 0.0, "max")
                # core
                if core_edges:
                    lowc = (1 - mu_core) ** n
                    midc = sum(((1 - mass_of(K, masses)) ** n for e in core_edges for K in sup.mincov[e]), F(0))
                    upc = sum((len(sup.mincov[e]) * (1 - mu[e]) ** n for e in core_edges), F(0))
                    ok_c = lowc <= err_c <= midc <= upc
                    upd("min_err_over_lower_core", float(err_c / lowc), "min")
                    upd("max_err_over_upper_core", float(err_c / midc) if midc else 0.0, "max")
                else:
                    ok_c = err_c == 0
                    if err_c != 0:
                        stats["empty_core_nonzero_error"] += 1
                if not (ok_f and ok_c):
                    stats["k1_violations"] += 1
                    record("k1_bounds", dict(S=S, lab=list(lab.values()), masses=[str(m) for m in masses], n=n,
                                             err_fam=str(err_f), err_core=str(err_c)))
                if brute_budget[0] > 0 and n <= 4 and sup.k ** n <= 625 and rng.random() < 0.02:
                    brute_budget[0] -= 1
                    stats["brute_crosschecks"] += 1
                    if brute_error(sup.ind_f, masses, n) != err_f or brute_error(sup.ind_c, masses, n) != err_c:
                        stats["brute_mismatches"] += 1
                        record("mobius_vs_brute", dict(S=S, n=n))

            # K2 exact TV and two-point sanity with the plug-in, on minimum covers
            for e in sup.clutter:
                for K in sup.mincov[e]:
                    mK = mass_of(K, masses)
                    if mK != mu[e]:
                        continue
                    Q = sub_support(K)
                    keep_idx = [i for i in range(sup.k) if not K >> i & 1]
                    qm = [masses[i] / (1 - mK) for i in keep_idx]
                    tv = (sum((abs(masses[i] - (masses[i] / (1 - mK) if i in keep_idx else 0)) for i in range(sup.k)), F(0))) / 2
                    if tv != mK:
                        stats["k2_violations"] += 1
                        record("k2_tv", dict(S=S, e=sorted(e), K=K, tv=str(tv), mK=str(mK)))
                    for n in range(1, nmax + 1):
                        stats["k2_twopoint_checks"] += 1
                        if len(e) == 1:
                            eP = exact_error(sup.c_core, masses, n)
                            eQ = exact_error(Q.c_core, qm, n)
                        else:
                            eP = exact_error(sup.c_fam, masses, n)
                            eQ = exact_error(Q.c_fam, qm, n)
                        if eP + eQ < (1 - mK) ** n:
                            stats["k2_twopoint_violations"] += 1
                            record("k2_twopoint", dict(S=S, e=sorted(e), K=K, n=n))

            # K3 exact estimator errors (first mass vector only)
            if kind == kinds[0] and sup.core:
                for e in sup.clutter:
                    for K in sup.mincov[e]:
                        cp = all(any(not (K >> i & 1) and not (K >> j & 1) for i, j in sup.pairs[frozenset([b])])
                                 for b in sup.core)
                        if not cp:
                            continue
                        Q = sub_support(K)
                        mK = mass_of(K, masses)
                        keep_q = [i for i in range(sup.k) if not K >> i & 1]
                        qm = [masses[i] / (1 - mK) for i in keep_q]
                        for a in sup.core:
                            for Ka in sup.mincov[frozenset([a])]:
                                A = Ka & ~K
                                if A == 0:
                                    continue
                                R = sub_support(Ka)
                                # estimator on P's atom indices
                                indP = [False] * (1 << sup.k)
                                for T in range(1, 1 << sup.k):
                                    out_core = R.core if (T & A) == 0 else sup.core
                                    indP[T] = out_core != sup.core
                                A_q = sum(1 << keep_q.index(i) for i in members(A, sup.k))
                                indQ = [False] * (1 << Q.k)
                                for T in range(1, 1 << Q.k):
                                    out_core = R.core if (T & A_q) == 0 else sup.core
                                    indQ[T] = out_core != Q.core
                                keep_r = [i for i in range(sup.k) if not Ka >> i & 1]
                                rm = [masses[i] / (1 - mass_of(Ka, masses)) for i in keep_r]
                                cP = mobius_coeffs(indP, sup.k)
                                cQ = mobius_coeffs(indQ, Q.k)
                                pA = mass_of(A, masses)
                                qA = sum((qm[keep_q.index(i)] for i in members(A, sup.k)), F(0))
                                for n in range(1, nmax + 1):
                                    stats["k3_checks"] += 1
                                    eP = exact_error(cP, masses, n)
                                    eQ = exact_error(cQ, qm, n)
                                    # at R no atom of K_a has mass, so the estimator always outputs CORE_R
                                    okay = (eP == (1 - pA) ** n and eQ == (1 - qA) ** n and eQ <= (1 - pA) ** n)
                                    if not okay:
                                        stats["k3_violations"] += 1
                                        record("k3_estimator", dict(S=S, e=sorted(e), K=K, a=a, Ka=Ka, n=n,
                                                                    eP=str(eP), eQ=str(eQ)))
        stats.pop("_k3_pending", None)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=20260915)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    t0 = time.time()
    out = dict(seed=args.seed,
               smoke=args.smoke, configs=[])
    brute_budget = [400]
    if args.smoke:
        cfgs = [("smoke_p3_binary", 3, (0, 1), 2, 3, ["uniform", "rare"], 5)]
    else:
        cfgs = [("p3_binary", 3, (0, 1), 2, 5, ["uniform", "uniform", "rare"], 8),
                ("p4_binary", 4, (0, 1), 2, 4, ["uniform"], 6),
                ("p3_multiclass", 3, (0, 1, 2), 2, 4, ["uniform"], 6)]
    for cfg in cfgs:
        t = time.time()
        st = run_config(*cfg, rng=rng, out=out, brute_budget=brute_budget)
        st["seconds"] = round(time.time() - t, 1)
        out["configs"].append(st)
        print(json.dumps(st), flush=True)
    total_viol = sum(c["k1_violations"] + c["k2_violations"] + c["k2_twopoint_violations"] + c["k3_violations"]
                     + c["lemma1_iii_violations"] + c["ordering_violations"] + c["empty_core_nonzero_error"]
                     + c["brute_mismatches"] for c in out["configs"])
    out["total_violations"] = total_viol
    out["verdict"] = "PASS" if total_viol == 0 else "FAIL"
    out["seconds"] = round(time.time() - t0, 1)
    name = "k1_k3_smoke.json" if args.smoke else "k1_k3_results.json"
    with open(os.path.join(HERE, name), "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("VERDICT", out["verdict"], "violations", total_viol, "seconds", out["seconds"])


if __name__ == "__main__":
    main()
