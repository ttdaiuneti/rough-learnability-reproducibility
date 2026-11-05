# Finite empirical populations (Section 7.2)

Screening, exact structure computation, and sampling-with-replacement recovery curves on the
finite empirical populations built from public UCI datasets.

## Layout

| File | What it is |
|---|---|
| `exact_rough.py` | exact discernibility structure, clutter, reduct family and core; two independent routes, cross-checked |
| `validation.py` | route-equivalence and recovery-characterisation self-tests |
| `run_screening.py` | downloads and preprocesses the 20 candidate datasets, applies the realisability (Assumption 2) and feasibility gates |
| `run_population_truth.py` | records the exact structure of every accepted population (Table 4) |
| `run_recovery.py` | IID sampling with replacement; recovery thresholds and the Table 4 summary |
| `run_certificate_analysis.py` | minimal recovery certificates (Section 7.2, "Certificates") |
| `data/raw/` | the downloaded UCI files, cached |
| `data/processed/` | preprocessed tables, metadata, and recorded population truth, one triple per accepted dataset |
| `results/` | raw aggregate CSVs, as reported in the paper |
| `protocol/` | the screening plan, preprocessing specification and computational budget, fixed before any data were downloaded |
| `data/hashes/` | SHA-256 checksums: `prereg.json` records the three protocol documents at the time they were frozen (2026-09-12T05:54:08Z); `raw.json` records the downloaded UCI files; `processed.json` and `truth.json` record the processed tables and the population truths |

## Run

```
pip install -r requirements.txt
python3 run_screening.py --offline   # uses the cached files in data/raw/; drop --offline to re-download
python3 run_population_truth.py
python3 validation.py
python3 run_recovery.py
python3 run_certificate_analysis.py
```

All seven accepted populations are already screened and their structure recorded in
`data/processed/`; `run_screening.py --offline` reproduces this from the cached raw files
without any network access. `run_recovery.py` reproduces `results/recovery_results.csv`,
`threshold_results.csv` and `recovery_summary.csv` (Table 4); it accepts `--reps` and
`--reps-refine` to control the Monte Carlo budget, and defaults to the values used for the
reported numbers.

Standard scientific Python only (numpy, pandas); no rough-set package is used, every structural
computation in `exact_rough.py` is exact enumeration over attribute bitmasks.

## Data provenance

The datasets are the public UCI Machine Learning Repository sources listed in each
`data/processed/<slug>.meta.json` (`sources` field); the paper cites the repository as a whole. Twenty candidates were screened; thirteen were excluded (six for violating
realisability, five for exceeding the attribute-count budget, one for too few rows, one for
exceeding the exact-computation budget) and seven were accepted, as reported in
`results/screening_results.csv`.
