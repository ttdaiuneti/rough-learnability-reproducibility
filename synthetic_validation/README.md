# Synthetic validation (Section 5.1 and Appendix B)

Exact and Monte Carlo checks of the closed-form results on the three-population construction
of Section 4.1 (`P_0`, `P_1`, `P_2`), plus the deterministic sample-complexity bounds of
Theorem 9.

## Layout

| File | What it is |
|---|---|
| `rough_exact.py` | exact empirical reduct family and core by enumeration; no heuristic search |
| `analytic_formulas.py` | closed-form probabilities stated in the paper, and the plug-in estimator's exact recovery probability derived by enumeration |
| `validation.py` | analytic identities, implementation identities, Monte Carlo tolerance rule |
| `run_core_family_recovery.py` | Appendix B.1: finite-sample recovery, Monte Carlo vs analytic |
| `run_rare_event_mechanism.py` | Appendix B.2: rare-atom waiting time and the structural transition |
| `run_sample_complexity_bounds.py` | Appendix B.3: deterministic sample-complexity separation (Table 3) |
| `results/` | raw aggregate CSVs, as reported in the paper, with their `SHA256SUMS` |
| `protocol/experiment_protocol.md` | the experiment protocol (grids, seeds, repetition counts, tolerances, re-run rule), written before the runs |

## Run

```
pip install -r requirements.txt
python3 validation.py
python3 run_sample_complexity_bounds.py
python3 run_core_family_recovery.py
python3 run_rare_event_mechanism.py
```

`validation.py` reproduces the analytic and implementation self-test (240 floating-point
identities, 54 exact-rational identities, 4000 lookup-vs-literal comparisons, all with zero
failures). The three `run_*.py` scripts reproduce `results/*.csv`; each accepts `--reps` (and
similar) flags to control the Monte Carlo budget, and defaults to the values used for the
reported numbers.

Standard scientific Python only (numpy, pandas); no rough-set package is used or needed, every
structural computation is exact enumeration in `rough_exact.py`.

The protocol refers to the scripts by their development names: `run_e0.py` is
`run_core_family_recovery.py` (Appendix B.1), `run_e05.py` is `run_rare_event_mechanism.py`
(Appendix B.2), and `run_e1.py` is `run_sample_complexity_bounds.py` (Appendix B.3).

The protocol mentions a validation report listing tolerance violations. The only violations were
the two rows at `eps = 0.2`, `n = 10` under `P_1` (one event counted for both targets); they are
kept in `results/core_family_recovery.csv` together with their re-run rows (`stage = rerun`) and
are described in Appendix B.1 of the paper.
