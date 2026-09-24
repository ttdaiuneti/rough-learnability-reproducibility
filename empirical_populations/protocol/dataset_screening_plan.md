# E2 dataset screening plan — PRE-REGISTERED, FROZEN

**Written:** 2026-09-12, **before** any recovery curve, any population truth, and any
download. Its SHA-256 is recorded in `reports/anti_cherry_picking_audit.md` together with the
timestamp; `run_screening.py` re-hashes this file at run time and refuses to proceed if it
changed.

The point of this document is that the accept/exclude decision for every dataset is a
mechanical consequence of rules fixed here, and of nothing observed later. No rule below
mentions core recovery, family recovery, separation size, reduct count, or accuracy.

## 1. Candidate sources

1. UCI Machine Learning Repository, over HTTPS (reachable from this machine; verified 200).
2. OpenML, via `sklearn.datasets.fetch_openml`, used only when the UCI copy is awkward to
   parse.
3. `~/Documents/research-commons/datasets/` — checked first per the project's standing E1 rule.
   **It is empty** (README only), so it contributes no candidate. Nothing is added to it here;
   that requires an explicit request from the user.

## 2. Candidate list — closed, fixed now

The list is *closed*: it is written down before any download and may not be extended after
seeing results (protocol §29). It was assembled by one objective criterion — classification
decision tables that are natively categorical/nominal and are standard in the rough-set and
symbolic-learning literature — and deliberately spans both small and large `p`, including
datasets that the frozen limits below will certainly exclude.

| # | Dataset | Source | Expected N | Expected p |
|---|---|---|---|---|
| 1 | lenses | UCI | 24 | 4 |
| 2 | balloons (adult+stretch) | UCI | 20 | 4 |
| 3 | balance-scale | UCI | 625 | 4 |
| 4 | hayes-roth | UCI | 132 | 4 |
| 5 | tae (teaching assistant evaluation) | UCI | 151 | 5 |
| 6 | shuttle-landing-control | UCI | 15 | 6 |
| 7 | car evaluation | UCI | 1728 | 6 |
| 8 | monks-1 | UCI | 556 | 6 |
| 9 | monks-2 | UCI | 601 | 6 |
| 10 | monks-3 | UCI | 554 | 6 |
| 11 | nursery | UCI | 12960 | 8 |
| 12 | post-operative patient | UCI | 90 | 8 |
| 13 | breast-cancer (Ljubljana) | UCI | 286 | 9 |
| 14 | tic-tac-toe endgame | UCI | 958 | 9 |
| 15 | solar-flare (data2) | UCI | 1066 | 10 |
| 16 | zoo | UCI | 101 | 16 |
| 17 | house-votes-84 | UCI | 435 | 16 |
| 18 | primary-tumor | UCI | 339 | 17 |
| 19 | lymphography | UCI | 148 | 18 |
| 20 | spect heart (train+test) | UCI | 267 | 22 |

Entries 16–20 are expected to fail the frozen `p ≤ 15` budget. They are listed on purpose: a
candidate list containing only datasets that pass would itself be evidence of selection.

## 3. Stopping rule for dataset collection

The screen runs the closed list above exactly once, in the order given. It stops when the list
is exhausted. It does not stop early because enough datasets have been accepted, and it is not
extended if few are accepted. If a download fails for network reasons, the dataset is marked
`EXCLUDED_DOWNLOAD_FAILED` and the list is **not** topped up with a replacement.

## 4. Preprocessing rule

Full detail in `preprocessing_spec.md`; the rules that bear on inclusion are:

* drop columns that are identifiers or free-text names carrying no condition information
  (declared per dataset in `preprocessing_spec.md`, never inferred from results);
* missing values become their own category `"?"`, never imputed and never used to drop rows;
* natively categorical attributes are used as-is;
* continuous attributes, where present, are discretised by one fixed unsupervised procedure
  (equal-frequency 4 bins), identical across all datasets;
* duplicate rows are **kept** — they carry probability mass in `P_N`;
* no row IDs, no synthetic attributes, no relabeling.

## 5. Exclusion criteria — applied in this order

| Code | Condition |
|---|---|
| `EXCLUDED_DOWNLOAD_FAILED` | source unreachable or unparseable |
| `EXCLUDED_TOO_FEW_ROWS` | processed `N < 20` |
| `EXCLUDED_P_BUDGET` | processed `p > 15` (frozen engineering limit, `computational_budget.md`) |
| `EXCLUDED_A2_CONFLICT` | some identical full condition vectors carry different labels |
| `EXCLUDED_EXACT_FAMILY_INFEASIBLE` | exact population family exceeds the frozen budget |
| `EXCLUDED_DEGENERATE_STRUCTURE` | no bichromatic pair exists (single class after processing), so the reduct family is `{∅}` and the identification question is vacuous |

A2 is checked on the processed condition vector, exactly as protocol §3 requires. The forbidden
repairs (majority vote, dropping conflicting rows, adding IDs, VPRS, stochastic labels) are not
implemented anywhere in this stage; `run_screening.py` contains no code path that modifies
labels.

Every excluded dataset is reported with its code and, for A2, the number of conflicting
condition vectors and the number of rows involved.

## 6. What counts as accepted

A dataset is accepted iff it passes every criterion in §5. Acceptance is independent of:
`|R_{P_N}|`, core size, whether any separation appears, and any downstream number. Accepted
populations are then partitioned into Group U (`|R_{P_N}| = 1`) and Group M (`|R_{P_N}| > 1`)
**before** recovery analysis, per protocol §9.

## 7. Target count

Roughly 10–20 candidates, as the protocol suggests: the list has 20. The number of *accepted*
datasets is whatever the rules produce — it is not a target, and a small accepted set is a
reportable outcome, not a reason to widen the list.

## 8. Order of operations (enforced by the scripts)

1. freeze this plan (hash recorded);
2. `run_screening.py` → download, process, A2 gate, feasibility gate → `screening_results.csv`;
3. `run_population_truth.py` → exact `R_{P_N}`, `CORE_{P_N}`, hashes → `population_truth.csv`;
4. only then `run_e2_recovery.py` → sampling, recovery curves;
5. `run_certificate_analysis.py` → experimental recovery certificates.

Steps 1–3 involve no recovery curve of any kind. The first sight of a recovery number is in
step 4.
