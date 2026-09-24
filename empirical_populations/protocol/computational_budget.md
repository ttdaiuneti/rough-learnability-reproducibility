# E2 computational budget — PRE-REGISTERED, FROZEN

Fixed before any download, any population truth, and any recovery curve. These are engineering
limits, not theorems (protocol §7). A dataset that exceeds one of them is excluded and reported;
it is never quietly downgraded to a heuristic.

## 1. Exact population structure

| Limit | Value | Exclusion code if exceeded |
|---|---|---|
| condition attributes `p` | ≤ 15 | `EXCLUDED_P_BUDGET` |
| distinct labeled atoms (support size) | ≤ 5 000 | `EXCLUDED_EXACT_FAMILY_INFEASIBLE` |
| bichromatic atom pairs | ≤ 2 000 000 | `EXCLUDED_EXACT_FAMILY_INFEASIBLE` |
| wall-clock per full-population exact family | ≤ 300 s | `EXCLUDED_EXACT_FAMILY_INFEASIBLE` |
| peak memory per dataset | ≤ 4 GB | `EXCLUDED_EXACT_FAMILY_INFEASIBLE` |
| hard cap on number of reducts reported | none | — |

`p ≤ 15` makes the minimal-transversal step a brute-force sweep over at most `2^15 = 32 768`
attribute subsets, which is exact by construction and costs milliseconds. No output cap on the
reduct family is needed at this `p`, so none is imposed: the *complete* family is always
reported (protocol §6).

## 2. Sampling experiment

| Limit | Value |
|---|---|
| Monte Carlo repetitions `N_REP` | 10 000 |
| repetitions at threshold-critical points | 100 000 |
| wall-clock per dataset for the full recovery grid | ≤ 1 800 s |
| sample-size grid | `{0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5} × N`, integer-clipped to `≥ 1`, unioned with the log ladder `{1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000}` restricted to `≤ 5N`, deduplicated and sorted |

Sampling is IID **with replacement** from the frozen `P_N`, so `n > N` is meaningful and the
grid deliberately runs to `5N`.

If a dataset cannot complete the grid at `N_REP = 10 000` inside the wall-clock limit, the
protocol's fallback applies: a pilot is run **without inspecting separation conclusions**, a
lower repetition count justified by binomial precision is frozen for that dataset, and the
reduced count is recorded in every affected row of `recovery_results.csv` and in the reports.

## 3. Threshold convention (frozen before curves are seen)

Confidence target `q = 0.95`, reported in two forms, never merged:

* **point-estimate threshold** — the smallest grid `n` with `p̂(n) ≥ 0.95`;
* **confidence-certified threshold** — the smallest grid `n` whose Wilson 95 % *lower* bound is
  `≥ 0.95`.

If neither is attained on the frozen grid, the value is right-censored and reported as
`> n_max`. Censored entries are never extrapolated and never dropped; the descriptive ratio
`S = n_family^0.95 / n_core^0.95` is computed only when both thresholds are observed, and
censoring is preserved otherwise (protocol §14).

## 4. Certificate enumeration

| Limit | Value |
|---|---|
| max minimal certificates enumerated per target | 200 000 |
| max wall-clock per dataset | ≤ 600 s |
| behaviour on overrun | report `CERTIFICATE_ENUMERATION_INFEASIBLE`, never approximate silently |

## 5. Determinism

Every random draw comes from `numpy.random.default_rng(SeedSequence([...]))` with an integer key
built from the dataset index, the grid index and the repetition count; the literal key is stored
in the result rows so any single point can be re-run in isolation.
