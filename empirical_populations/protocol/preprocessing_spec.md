# E2 preprocessing specification — PRE-REGISTERED, FROZEN

Fixed before any download. One procedure, applied identically to every candidate. No variant is
tried and no choice below may be revisited after seeing recovery curves (protocol §5, §19-NC5).

## 1. Column handling

| Column kind | Rule |
|---|---|
| condition attributes, categorical/nominal | used as-is, values mapped to integer codes by sorted order of the raw string |
| condition attributes, continuous | discretised (see §3) |
| decision attribute | the dataset's standard class column, mapped to integer codes; never modified, never merged, never majority-voted |
| identifier / free-text name | dropped — carries no condition information and would make every row its own equivalence class |

A column is an identifier iff it is unique across all rows and is either a running integer index
or a free-text name. Identifier columns, alternative decision columns, and continuous columns
are declared per dataset **in advance**, in the frozen table of §8.

### Amendment 1 — 2026-09-12, before any download

The first draft named identifier columns only for `hayes-roth` and `zoo`. While writing the
loader it became clear that `lenses.data` carries a leading running index, that the `monks-*`
files carry a trailing `data_N` tag, and that `solar-flare` data2 carries three alternative
decision counts of which only one can be the decision. Leaving a running index in place would be
exactly the forbidden repair of protocol §3 — an added row ID makes every row its own
equivalence class and forces artificial realizability — so the declarations are completed in §8
below.

This amendment is recorded rather than silently applied. It was made **before any file was
downloaded**, therefore before any population truth and before any recovery curve existed; it
cannot have been influenced by results. The hash of this file as amended is what
`run_screening.py` freezes and re-checks on every later run.

Dropping an identifier is the opposite of the forbidden repair in protocol §3: adding row IDs
would force artificial realizability, so a column that *is* a row ID must go before the A2 gate,
not after it.

## 2. Missing values

One rule, everywhere: **a missing value becomes its own category**, coded from the literal token
used by the source (`?` in the UCI files).

Rationale, and why the alternatives are excluded:

* imputation would invent condition values and change the finite population `P_N`;
* dropping rows with missing values could *create* realizability by deleting exactly the rows
  that conflict, which protocol §3 forbids as a repair — even done innocently, it would make the
  A2 gate meaningless.

Missing-as-category is therefore both the honest option and the one that leaves the A2 gate with
teeth.

## 3. Discretisation of continuous attributes

Preference is for natively categorical tables; on the frozen candidate list the only datasets
with continuous or ordinal-numeric columns are `tae` (one count-valued column) and
`solar-flare` (small integer counts).

Where a continuous attribute occurs, the fixed procedure is:

> **equal-frequency binning into 4 bins**, computed from that attribute's marginal distribution
> in the processed dataset, ties broken by `pandas.qcut(..., duplicates="drop")`; if fewer than
> 4 distinct bin edges survive, the attribute keeps its distinct values as categories.

This procedure is **unsupervised**: it never reads the decision column, and it therefore cannot
use core recovery, family recovery, reduct count, separation, or classification accuracy as a
tuning target — none of those quantities exists at preprocessing time. The bin count 4 is fixed
here and is not varied.

An attribute with at most 10 distinct integer values is treated as already categorical and is
not binned; this threshold is fixed now and applied uniformly.

## 4. Duplicate rows

Kept. Duplicates are not redundant here: `P_N(x, y)` is defined by row multiplicity (protocol
§2), so removing duplicates would change the population, its atom masses, and therefore every
sampling result. De-duplication happens only when forming the *support* (the distinct labeled
atoms), with the multiplicities retained as masses.

## 5. Row order

Irrelevant to every quantity computed in E2, but rows are kept in source order and the processed
file is written deterministically so that its hash is stable.

## 6. Output

For each processed dataset, `data/processed/<slug>.csv` with integer-coded columns
`a0 … a{p-1}` and `y`, plus `data/processed/<slug>.meta.json` recording: source URL, original
`N`/`p`, processed `N`/`p`, dropped columns and why, per-column value maps, which columns were
discretised, and the count of missing tokens per column. The SHA-256 of both files goes to
`data/hashes/`.

The value maps make the encoding reversible, which matters because a reviewer must be able to
check that no relabeling happened.

## 8. Frozen per-dataset declarations

Column indices are 0-based positions in the raw file. `class` is the decision column; `drop` are
identifier or alternative-decision columns; `numeric` are the columns subject to the §3
equal-frequency binning. Everything not listed as `numeric` is treated as nominal, however many
levels it has — an instructor id with 25 levels is a nominal code, not a quantity.

| Dataset | sep | class | drop | numeric | missing token |
|---|---|---|---|---|---|
| lenses | whitespace | 5 | 0 (running index) | — | — |
| balloons | `,` | 4 | — | — | — |
| balance-scale | `,` | 0 | — | — | — |
| hayes-roth | `,` | 5 | 0 (`name`) | — | — |
| tae | `,` | 5 | — | 4 (class size, a count) | — |
| shuttle-landing-control | `,` | 0 | — | — | `*` |
| car | `,` | 6 | — | — | — |
| monks-1/2/3 | whitespace | 0 | 7 (`data_N` tag) | — | — |
| nursery | `,` | 8 | — | — | — |
| post-operative | `,` | 8 | — | — | `?` |
| breast-cancer | `,` | 0 | — | — | `?` |
| tic-tac-toe | `,` | 9 | — | — | — |
| solar-flare | whitespace | 10 (C-class count) | 11, 12 (M- and X-class counts: alternative decisions, not conditions) | — | — |
| zoo | `,` | 17 | 0 (`animal_name`) | — | — |
| house-votes-84 | `,` | 0 | — | — | `?` |
| primary-tumor | `,` | 0 | — | — | `?` |
| lymphography | `,` | 0 | — | — | — |
| spect | `,` | 0 | — | — | — |

`monks-*` uses `train` and `test` concatenated: together they are the dataset, and duplicate
rows across the two files are kept because they carry mass in `P_N`.

## 7. What is never done

No normalisation or scaling (meaningless for exact Pawlak structures, and E2 computes no
distances). No feature selection. No class balancing. No sampling at preprocessing time. No
attribute construction. No use of `research-commons/eval_harness/protocol.py`: that harness
evaluates reduced attribute sets with 3-NN and SVM-RBF, and protocol §21 forbids classifier
outcomes in E2 — there is no evaluation step in this stage to route through it.
