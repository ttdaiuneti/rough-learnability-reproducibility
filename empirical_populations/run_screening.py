"""Screening -- download, process, and gate the closed candidate list of UCI datasets (Section 5.2).

Runs the closed candidate list below exactly once, in order. It computes NO recovery curve:
the first sight of a recovery number is in run_recovery.py, two steps later.

Gates, applied in this order:
    download -> too few rows -> p budget -> realisability (Assumption 2)
    -> exact-family feasibility -> degenerate structure

Usage: python3 run_screening.py [--offline]
"""
import argparse
import hashlib
import io
import json
import os
import time
import urllib.request

import pandas as pd

import exact_rough as er

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "raw")
PROC = os.path.join(HERE, "data", "processed")
HASHES = os.path.join(HERE, "data", "hashes")
RESULTS = os.path.join(HERE, "results")

# Computational budget (Appendix A of the paper).
MAX_P = 15
MAX_SUPPORT = 5000
MAX_PAIRS = 2_000_000
MAX_TRUTH_SECONDS = 300
MIN_ROWS = 20

UCI = "https://archive.ics.uci.edu/ml/machine-learning-databases"

# The closed candidate list: 20 UCI datasets, with the column declarations used to preprocess
# each one (decision column `cls`, dropped columns, separator, missing-value tokens, and any
# numeric column binned into quartiles before being treated as categorical).
CANDIDATES = [
    dict(slug="lenses", urls=[f"{UCI}/lenses/lenses.data"], sep=r"\s+", cls=5, drop=[0]),
    dict(slug="balloons", urls=[f"{UCI}/balloons/adult+stretch.data"], sep=",", cls=4, drop=[]),
    dict(slug="balance-scale", urls=[f"{UCI}/balance-scale/balance-scale.data"], sep=",",
         cls=0, drop=[]),
    dict(slug="hayes-roth", urls=[f"{UCI}/hayes-roth/hayes-roth.data"], sep=",", cls=5,
         drop=[0]),
    dict(slug="tae", urls=[f"{UCI}/tae/tae.data"], sep=",", cls=5, drop=[], numeric=[4]),
    dict(slug="shuttle-landing-control",
         urls=[f"{UCI}/shuttle-landing-control/shuttle-landing-control.data"], sep=",", cls=0,
         drop=[], na=["*"]),
    dict(slug="car", urls=[f"{UCI}/car/car.data"], sep=",", cls=6, drop=[]),
    dict(slug="monks-1", urls=[f"{UCI}/monks-problems/monks-1.train",
                               f"{UCI}/monks-problems/monks-1.test"], sep=r"\s+", cls=0,
         drop=[7]),
    dict(slug="monks-2", urls=[f"{UCI}/monks-problems/monks-2.train",
                               f"{UCI}/monks-problems/monks-2.test"], sep=r"\s+", cls=0,
         drop=[7]),
    dict(slug="monks-3", urls=[f"{UCI}/monks-problems/monks-3.train",
                               f"{UCI}/monks-problems/monks-3.test"], sep=r"\s+", cls=0,
         drop=[7]),
    dict(slug="nursery", urls=[f"{UCI}/nursery/nursery.data"], sep=",", cls=8, drop=[]),
    dict(slug="post-operative", urls=[f"{UCI}/postoperative-patient-data/post-operative.data"],
         sep=",", cls=8, drop=[], na=["?"]),
    dict(slug="breast-cancer", urls=[f"{UCI}/breast-cancer/breast-cancer.data"], sep=",", cls=0,
         drop=[], na=["?"]),
    dict(slug="tic-tac-toe", urls=[f"{UCI}/tic-tac-toe/tic-tac-toe.data"], sep=",", cls=9,
         drop=[]),
    dict(slug="solar-flare", urls=[f"{UCI}/solar-flare/flare.data2"], sep=r"\s+", cls=10,
         drop=[11, 12], skiprows=1),
    dict(slug="zoo", urls=[f"{UCI}/zoo/zoo.data"], sep=",", cls=17, drop=[0]),
    dict(slug="house-votes-84", urls=[f"{UCI}/voting-records/house-votes-84.data"], sep=",",
         cls=0, drop=[], na=["?"]),
    dict(slug="primary-tumor", urls=[f"{UCI}/primary-tumor/primary-tumor.data"], sep=",", cls=0,
         drop=[], na=["?"]),
    dict(slug="lymphography", urls=[f"{UCI}/lymphography/lymphography.data"], sep=",", cls=0,
         drop=[]),
    dict(slug="spect", urls=[f"{UCI}/spect/SPECT.train", f"{UCI}/spect/SPECT.test"], sep=",",
         cls=0, drop=[]),
]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- loading
def fetch(url, offline):
    os.makedirs(RAW, exist_ok=True)
    dest = os.path.join(RAW, url.rsplit("/", 1)[-1])
    if not os.path.exists(dest):
        if offline:
            raise FileNotFoundError(f"offline and not cached: {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "screening/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        with open(dest, "wb") as fh:
            fh.write(data)
    return dest


def load_raw(cand, offline):
    frames = []
    for url in cand["urls"]:
        path = fetch(url, offline)
        with open(path, "r", errors="replace") as fh:
            text = fh.read()
        frames.append(pd.read_csv(io.StringIO(text), sep=cand["sep"], header=None,
                                  engine="python", skiprows=cand.get("skiprows", 0),
                                  skip_blank_lines=True, comment=None))
    df = pd.concat(frames, ignore_index=True)
    return df.dropna(how="all")


def process(cand, raw):
    """Apply the preprocessing rules declared in CANDIDATES. Returns (frame, meta)."""
    na = cand.get("na", [])
    df = raw.copy()
    df.columns = list(range(df.shape[1]))
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()
    n_missing = {int(c): int(df[c].isin(na).sum()) for c in df.columns} if na else {}
    # missing values become their own category: nothing is imputed and no row is dropped
    cls = cand["cls"]
    drop = set(cand.get("drop", []))
    cond_cols = [c for c in df.columns if c != cls and c not in drop]
    numeric = set(cand.get("numeric", []))

    binned = []
    for c in cond_cols:
        if c in numeric:
            vals = pd.to_numeric(df[c], errors="coerce")
            try:
                b = pd.qcut(vals, 4, duplicates="drop")
                df[c] = b.astype(str)
                binned.append(int(c))
            except ValueError:
                pass  # too few distinct edges: keep the raw values as categories

    maps, out = {}, {}
    for k, c in enumerate(cond_cols):
        cats = sorted(df[c].unique())
        maps[f"a{k}"] = {v: i for i, v in enumerate(cats)}
        out[f"a{k}"] = df[c].map(maps[f"a{k}"]).to_numpy()
    ycats = sorted(df[cls].unique())
    maps["y"] = {v: i for i, v in enumerate(ycats)}
    out["y"] = df[cls].map(maps["y"]).to_numpy()

    frame = pd.DataFrame(out)
    meta = dict(slug=cand["slug"], sources=cand["urls"], original_N=int(len(raw)),
                original_p=int(raw.shape[1] - 1), processed_N=int(len(frame)),
                processed_p=int(len(cond_cols)), n_classes=len(ycats),
                dropped_columns=sorted(drop), decision_column=cls,
                discretised_columns=binned, missing_token_counts=n_missing,
                value_maps={k: {str(a): int(b) for a, b in v.items()} for k, v in maps.items()})
    return frame, meta


# --------------------------------------------------------------------------- screening
def screen_one(cand, offline):
    row = dict(dataset=cand["slug"], source=cand["urls"][0], original_N=None, processed_N=None,
               original_p=None, processed_p=None, classes=None, preprocessing="fixed spec",
               a2_status=None, a2_conflict_vectors=None, a2_conflict_rows=None,
               feasibility=None, support=None, bichromatic_pairs=None,
               truth_seconds=None, status=None, exclusion_reason="")
    try:
        raw = load_raw(cand, offline)
    except Exception as exc:
        row.update(status="EXCLUDED", exclusion_reason=f"EXCLUDED_DOWNLOAD_FAILED: {exc}")
        return row, None, None

    frame, meta = process(cand, raw)
    row.update(original_N=meta["original_N"], original_p=meta["original_p"],
               processed_N=meta["processed_N"], processed_p=meta["processed_p"],
               classes=meta["n_classes"])

    if meta["processed_N"] < MIN_ROWS:
        row.update(status="EXCLUDED", exclusion_reason="EXCLUDED_TOO_FEW_ROWS")
        return row, frame, meta
    if meta["processed_p"] > MAX_P:
        row.update(status="EXCLUDED", exclusion_reason="EXCLUDED_P_BUDGET")
        return row, frame, meta

    X = frame[[f"a{k}" for k in range(meta["processed_p"])]].to_numpy()
    y = frame["y"].to_numpy()
    pop = er.Population(X, y)
    nconf, nrows, examples = pop.realisability_conflicts()
    row.update(a2_conflict_vectors=nconf, a2_conflict_rows=nrows,
               a2_status="OK" if nconf == 0 else "CONFLICT", support=len(pop.X))
    if nconf:
        row.update(status="EXCLUDED", exclusion_reason="EXCLUDED_REALISABILITY_CONFLICT")
        meta["realisability_examples"] = examples
        return row, frame, meta

    npairs = pop.n_bichromatic_pairs()
    row["bichromatic_pairs"] = npairs
    if len(pop.X) > MAX_SUPPORT or npairs > MAX_PAIRS:
        row.update(status="EXCLUDED", feasibility="OVER_BUDGET",
                   exclusion_reason="EXCLUDED_EXACT_FAMILY_INFEASIBLE")
        return row, frame, meta
    if npairs == 0:
        row.update(status="EXCLUDED", exclusion_reason="EXCLUDED_DEGENERATE_STRUCTURE")
        return row, frame, meta

    t0 = time.perf_counter()
    summary = er.population_summary(pop)
    secs = time.perf_counter() - t0
    row["truth_seconds"] = round(secs, 3)
    if secs > MAX_TRUTH_SECONDS:
        row.update(status="EXCLUDED", feasibility="OVER_TIME",
                   exclusion_reason="EXCLUDED_EXACT_FAMILY_INFEASIBLE")
        return row, frame, meta
    row.update(status="ACCEPTED", feasibility="OK")
    meta["structure_probe"] = {k: v for k, v in summary.items() if k != "family"}
    return row, frame, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="use only cached raw files")
    args = ap.parse_args()
    os.makedirs(PROC, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(HASHES, exist_ok=True)

    rows, hashes = [], {}
    for cand in CANDIDATES:
        row, frame, meta = screen_one(cand, args.offline)
        rows.append(row)
        if frame is not None:
            cpath = os.path.join(PROC, f"{cand['slug']}.csv")
            mpath = os.path.join(PROC, f"{cand['slug']}.meta.json")
            frame.to_csv(cpath, index=False)
            with open(mpath, "w") as fh:
                json.dump(meta, fh, indent=2, sort_keys=True)
            hashes[f"{cand['slug']}.csv"] = sha256_file(cpath)
            hashes[f"{cand['slug']}.meta.json"] = sha256_file(mpath)
        print(f"  {row['dataset']:<26} {str(row['status']):<9} "
              f"N={row['processed_N']} p={row['processed_p']} "
              f"{row['exclusion_reason']}", flush=True)

    with open(os.path.join(HASHES, "processed.json"), "w") as fh:
        json.dump(hashes, fh, indent=2, sort_keys=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "screening_results.csv"), index=False)
    acc = df[df.status == "ACCEPTED"]
    print(f"\nscreened {len(df)} candidates, accepted {len(acc)}")
    print(df.exclusion_reason.replace("", "ACCEPTED").value_counts().to_string())


if __name__ == "__main__":
    main()
