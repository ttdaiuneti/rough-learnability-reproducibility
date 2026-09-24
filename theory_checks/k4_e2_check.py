"""Pre-registered check K4 of theory_spec.md on the FROZEN E2 populations.

Uses E2's own loader conventions and exact_rough.py (imported read-only from the frozen E2 folder);
no re-download, no re-preprocessing. Masses are P_N = multiplicity / N, kept as integer counts so
that cover masses are exact rationals.

K4 (a) Group U: every clutter edge is a singleton (so mu_fam = mu_core and the events coincide).
K4 (b) tic-tac-toe: CORE = empty.
K4 (c) tic-tac-toe: recorded n_family^{0.95} = 1916 is not below ceil(log 20 / -log(1 - mu_fam)).
Descriptive only (not pre-registered): for every recorded family grid row of tic-tac-toe, compare
the F1 lower bound (1 - mu_fam)^n on the error with the Wilson interval of the observed error.
"""
import csv
import hashlib
import json
import math
import os
import sys
from fractions import Fraction as F

import networkx as nx
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
E2 = os.path.normpath(os.path.join(HERE, "..", "empirical_populations"))
sys.path.insert(0, E2)
import exact_rough as er  # noqa: E402

ACCEPTED = {"lenses": "U", "balloons": "U", "balance-scale": "U", "car": "U",
            "monks-1": "U", "monks-2": "U", "tic-tac-toe": "M"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()


def load(name):
    path = os.path.join(E2, "data", "processed", name + ".csv")
    with open(path) as fh:
        rows = list(csv.reader(fh))
    header, body = rows[0], rows[1:]
    ycol = header.index("y")
    xcols = [i for i in range(len(header)) if i != ycol]
    X = np.array([[int(r[i]) for i in xcols] for r in body], dtype=np.int64)
    y = np.array([int(r[ycol]) for r in body], dtype=np.int64)
    return er.Population(X, y), sha256(path)


def cover_mass_binary(pop, e):
    """Exact minimum-weight vertex cover of the witness graph of edge e (binary labels), by min-cut
    with integer capacities = multiplicities. Returns (cover_count_mass, N)."""
    masks, i, j = pop.pair_edges()
    sel = masks == e
    ii, jj = i[sel], j[sel]
    labels = sorted(set(pop.y.tolist()))
    assert len(labels) == 2, "min-cut route is only valid for binary labels"
    G = nx.DiGraph()
    for a, b in zip(ii.tolist(), jj.tolist()):
        left, right = (a, b) if pop.y[a] == labels[0] else (b, a)
        G.add_edge("s", ("L", left), capacity=int(pop.counts[left]))
        G.add_edge(("R", right), "t", capacity=int(pop.counts[right]))
        G.add_edge(("L", left), ("R", right))  # infinite capacity
    cut, _ = nx.minimum_cut(G, "s", "t")
    return int(cut), int(pop.counts.sum()), int(sel.sum())


def main():
    out = dict(datasets={})
    ok_all = True
    for name, group in ACCEPTED.items():
        pop, h = load(name)
        clutter = [int(m) for m in pop.clutter]
        sizes = [bin(m).count("1") for m in clutter]
        fam = er.minimal_transversals(pop.clutter, pop.p)
        core = er.core_mask(fam)
        d = dict(group=group, N=pop.N, p=pop.p, support=len(pop.X), csv_sha256=h,
                 n_clutter=len(clutter), clutter_edge_sizes=sorted(set(sizes)),
                 n_labels=len(set(pop.y.tolist())), core_size=bin(core).count("1"),
                 n_reducts=len(fam))
        if group == "U":
            d["K4a_all_singletons"] = all(s == 1 for s in sizes)
            ok_all &= d["K4a_all_singletons"]
        else:
            d["K4b_core_empty"] = core == 0
            ok_all &= d["K4b_core_empty"]
            covers = []
            for e in clutter:
                c, N, npairs = cover_mass_binary(pop, e)
                covers.append(dict(edge=er.show_mask(e, pop.p), size=bin(e).count("1"),
                                   witness_pairs=npairs, min_cover_count=c))
            N = int(pop.counts.sum())
            cmin = min(c["min_cover_count"] for c in covers)
            mu = F(cmin, N)
            d["clutter_covers"] = sorted(covers, key=lambda r: r["min_cover_count"])
            d["mu_fam"] = str(mu)
            d["mu_fam_float"] = float(mu)
            rate = -math.log(1 - float(mu))
            bracket = math.ceil(math.log(20) / rate)
            d["K4c_lower_bracket_n"] = bracket
            rec = [r for r in csv.DictReader(open(os.path.join(E2, "results", "recovery_results.csv")))
                   if r["dataset"] == name and r["target"] == "family"]
            thr = [r for r in csv.DictReader(open(os.path.join(E2, "results", "threshold_results.csv")))
                   if r.get("dataset") == name]
            d["threshold_rows"] = thr
            d["K4c_recorded_threshold"] = 1916
            d["K4c_pass"] = 1916 >= bracket
            ok_all &= d["K4c_pass"]
            rows = []
            for r in rec:
                n = int(r["n"])
                err_lo = 1 - float(r["wilson_hi"])
                err_hi = 1 - float(r["wilson_lo"])
                lb = (1 - float(mu)) ** n
                rows.append(dict(n=n, stage=r["stage"], reps=int(r["reps"]), err_hat=1 - float(r["p_hat"]),
                                 err_wilson=[err_lo, err_hi], F1_lower=lb, consistent=lb <= err_hi))
            d["descriptive_rows"] = sorted(rows, key=lambda r: (r["n"], r["stage"]))
        out["datasets"][name] = d
        print(name, {k: v for k, v in d.items() if k not in ("clutter_covers", "descriptive_rows", "threshold_rows")}, flush=True)
    out["verdict"] = "PASS" if ok_all else "FAIL"
    with open(os.path.join(HERE, "k4_results.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("K4 VERDICT", out["verdict"])


if __name__ == "__main__":
    main()
