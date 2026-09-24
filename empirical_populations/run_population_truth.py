"""Record the exact Pawlak structure of every accepted population (Section 5.2, Table 4).

For every dataset passing the realisability screen and the feasibility budget, compute the
exact discernibility structure, the exact complete reduct family, and the exact core; record
the family size, the reduct-size distribution and the core size; then hash the processed
population and the truth file. Only after this may sampling begin (run_recovery.py).

Nothing in this script samples anything, and nothing here looks at recovery. Group U (unique
reduct) / M (multiple reducts) membership is assigned here, before any recovery analysis
exists.

Usage: python3 run_population_truth.py
"""
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd

import exact_rough as er

HERE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(HERE, "data", "processed")
HASHES = os.path.join(HERE, "data", "hashes")
RESULTS = os.path.join(HERE, "results")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    screening = pd.read_csv(os.path.join(RESULTS, "screening_results.csv"))
    accepted = screening[screening.status == "ACCEPTED"].dataset.tolist()
    os.makedirs(RESULTS, exist_ok=True)
    rows, hashes = [], {}

    for slug in accepted:
        cpath = os.path.join(PROC, f"{slug}.csv")
        frame = pd.read_csv(cpath)
        cols = [c for c in frame.columns if c.startswith("a")]
        pop = er.Population(frame[cols].to_numpy(), frame["y"].to_numpy())

        t0 = time.perf_counter()
        summary = er.population_summary(pop)
        secs = time.perf_counter() - t0

        core_t, fam_t = er.recovery_targets(pop)
        masses = pop.masses
        truth = dict(
            slug=slug, p=pop.p, N=pop.N, support=len(pop.X),
            clutter=[int(m) for m in pop.clutter],
            family=[int(m) for m in summary["family"]],
            core_mask=summary["core_mask"],
            core_target_edges=[int(m) for m in core_t],
            family_target_edges=[int(m) for m in fam_t],
            min_atom_mass=float(masses.min()), max_atom_mass=float(masses.max()),
            median_atom_mass=float(np.median(masses)))
        tpath = os.path.join(PROC, f"{slug}.truth.json")
        with open(tpath, "w") as fh:
            json.dump(truth, fh, indent=2, sort_keys=True)
        hashes[f"{slug}.csv"] = sha256_file(cpath)
        hashes[f"{slug}.truth.json"] = sha256_file(tpath)

        group = "U" if summary["n_reducts"] == 1 else "M"
        rows.append(dict(
            dataset=slug, N=pop.N, p=pop.p, support=len(pop.X),
            n_edges=summary["n_edges"], n_clutter=summary["n_clutter"],
            n_reducts=summary["n_reducts"], min_reduct=summary["min_reduct"],
            median_reduct=summary["median_reduct"], max_reduct=summary["max_reduct"],
            core_size=summary["core_size"], group=group,
            n_core_target_edges=len(core_t), n_family_target_edges=len(fam_t),
            min_atom_mass=truth["min_atom_mass"], median_atom_mass=truth["median_atom_mass"],
            truth_seconds=round(secs, 3)))
        print(f"  {slug:<26} N={pop.N:<7} p={pop.p:<3} support={len(pop.X):<6} "
              f"|R|={summary['n_reducts']:<5} core={summary['core_size']:<3} group {group}",
              flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "population_truth.csv"), index=False)
    with open(os.path.join(HASHES, "truth.json"), "w") as fh:
        json.dump(hashes, fh, indent=2, sort_keys=True)

    print(f"\nrecorded {len(df)} population truths")
    if len(df):
        print(f"  Group U (|R| = 1): {(df.group == 'U').sum()}")
        print(f"  Group M (|R| > 1): {(df.group == 'M').sum()}")


if __name__ == "__main__":
    main()
