"""Experimental recovery certificates for the finite empirical populations (Section 5.2).

The synthetic construction of Section 4.1 shows that exact recovery is equivalent to having
observed a specific set of support atoms. This script asks whether an analogous description
exists for the finite empirical populations.

Definition used here. Recovery of a target (core or complete family) holds iff every edge in
that target's edge set is realised, i.e. iff for every target edge some generating atom pair is
fully observed. A **minimal recovery certificate** is therefore a minimal set of support atoms
that contains, for at least one generating pair of every target edge, both of its atoms.
Minimal certificates are exactly the minimal elements of

    { union of the chosen pairs : one generating pair chosen per target edge }.

These are *experimental* certificates: a description of this specific population, not a new
theoretical object.

Feasibility. The enumeration above has `prod_e |pairs(e)|` combinations. That is tiny for some
populations and astronomically large for others, so a fixed budget applies: above the cap the
result is reported as CERTIFICATE_ENUMERATION_INFEASIBLE and nothing is approximated. There is
no partial enumeration reported as if it were complete, and no single favourable certificate
reported in place of the full set.

Usage: python3 run_certificate_analysis.py [--max-combinations 2000000]
"""
import argparse
import itertools
import json
import os
import time

import numpy as np
import pandas as pd

import exact_rough as er

HERE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(HERE, "data", "processed")
RESULTS = os.path.join(HERE, "results")

MAX_CERTIFICATES = 200_000
MAX_SECONDS = 600


def load(slug):
    frame = pd.read_csv(os.path.join(PROC, f"{slug}.csv"))
    cols = [c for c in frame.columns if c.startswith("a")]
    pop = er.Population(frame[cols].to_numpy(), frame["y"].to_numpy())
    with open(os.path.join(PROC, f"{slug}.truth.json")) as fh:
        truth = json.load(fh)
    return pop, truth


def minimal_sets(sets):
    """Inclusion-minimal elements of a collection of frozensets."""
    out = []
    for s in sorted(sets, key=len):
        if not any(k <= s for k in out):
            out.append(s)
    return out


def enumerate_certificates(pop, targets, max_combinations, deadline):
    """Exact enumeration, or (None, reason) when the combination budget is exceeded."""
    if len(targets) == 0:
        # nothing has to be realised: the empty observation already recovers the target
        return [frozenset()], "exact"
    masks, ii, jj = pop.pair_edges()
    pairs = []
    total = 1
    for e in targets:
        sel = masks == int(e)
        pl = list(zip(ii[sel].tolist(), jj[sel].tolist()))
        pairs.append(pl)
        total *= len(pl)
        if total > max_combinations:
            return None, (f"CERTIFICATE_ENUMERATION_INFEASIBLE: at least {total:.3g} "
                          f"combinations exceeds the cap {max_combinations:g}")
    seen = set()
    for combo in itertools.product(*pairs):
        if time.perf_counter() > deadline:
            return None, "CERTIFICATE_ENUMERATION_INFEASIBLE: wall-clock budget exceeded"
        u = frozenset(a for pr in combo for a in pr)
        seen.add(u)
        if len(seen) > MAX_CERTIFICATES:
            return None, (f"CERTIFICATE_ENUMERATION_INFEASIBLE: more than "
                          f"{MAX_CERTIFICATES} distinct unions")
    return minimal_sets(seen), "exact"


def describe(pop, certs):
    """Summarise a complete set of minimal certificates.

    The empty certificate is a real case, not a degenerate one to paper over: it occurs when the
    target has no edges to realise, so the target is recovered from any sample -- including the
    empty one. Rarity statistics are undefined for it (it requires no atom), so it is excluded
    from them and flagged separately instead of being folded in as a NaN."""
    sizes = [len(c) for c in certs]
    nonempty = [c for c in certs if c]
    rarest = [float(pop.masses[list(c)].min()) for c in nonempty]
    massed = [float(pop.masses[list(c)].sum()) for c in nonempty]
    return dict(n_certificates=len(certs),
                empty_certificate=any(len(c) == 0 for c in certs),
                min_card=min(sizes) if sizes else 0,
                max_card=max(sizes) if sizes else 0,
                median_card=float(np.median(sizes)) if sizes else 0.0,
                # "rarest atom mass in each certificate", summarised over certificates
                best_rarest_mass=max(rarest) if rarest else float("nan"),
                worst_rarest_mass=min(rarest) if rarest else float("nan"),
                min_total_mass=float(min(massed)) if massed else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-combinations", type=int, default=2_000_000)
    args = ap.parse_args()
    truth_tbl = pd.read_csv(os.path.join(RESULTS, "population_truth.csv"))
    rows = []
    for _, rec in truth_tbl.iterrows():
        slug = rec.dataset
        pop, truth = load(slug)
        row = dict(dataset=slug, group=rec.group, n_reducts=int(rec.n_reducts),
                   core_size=int(rec.core_size))
        store = {}
        for target, key in (("core", "core_target_edges"), ("family", "family_target_edges")):
            tgt = np.array(truth[key], dtype=np.int64)
            t0 = time.perf_counter()
            certs, status = enumerate_certificates(pop, tgt, args.max_combinations,
                                                   t0 + MAX_SECONDS)
            row[f"{target}_status"] = status
            row[f"{target}_seconds"] = round(time.perf_counter() - t0, 3)
            row[f"{target}_n_target_edges"] = len(tgt)
            if certs is None:
                for k in ("n_certificates", "min_card", "max_card", "median_card",
                          "best_rarest_mass", "worst_rarest_mass", "min_total_mass"):
                    row[f"{target}_{k}"] = np.nan
                store[target] = None
            else:
                d = describe(pop, certs)
                for k, v in d.items():
                    row[f"{target}_{k}"] = v
                store[target] = certs

        # does every family certificate need atoms rarer than those required by at least one
        # core certificate?
        if store["core"] is not None and store["family"] is not None:
            core_best = max((float(pop.masses[list(c)].min()) if c else 1.0)
                            for c in store["core"])
            fam_best = max((float(pop.masses[list(c)].min()) if c else 1.0)
                           for c in store["family"])
            row["family_needs_rarer_atoms"] = bool(fam_best < core_best)
            row["identical_certificate_sets"] = bool(
                set(store["core"]) == set(store["family"]))
        else:
            row["family_needs_rarer_atoms"] = np.nan
            row["identical_certificate_sets"] = np.nan
        rows.append(row)
        print(f"  {slug:<26} core: {row['core_status'][:44]:<44} "
              f"family: {row['family_status'][:44]}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "certificate_results.csv"), index=False)
    print(f"\nwrote results/certificate_results.csv ({len(df)} populations)")
    ok = df[df.family_status == "exact"]
    print(f"  exact family certificate enumeration: {len(ok)} of {len(df)} populations")


if __name__ == "__main__":
    main()
