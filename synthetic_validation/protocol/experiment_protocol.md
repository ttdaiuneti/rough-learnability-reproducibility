# E0/E1 experiment protocol

Frozen inputs: `../v2-gate/locked_spec_v2.md` and the passing gate report
`../v2-gate/gate_report.txt`. This stage validates and illustrates that theory. It does not
change it, and nothing here may be cited as a proof.

## 0. Scope

In scope: exact finite-sample recovery of the population core and of the complete population
reduct family on the frozen three-population class, and the sample-complexity separation
between the two targets.

Out of scope, by construction — none of these appears anywhere in the code or the reports:
classifier accuracy, predictive performance, reduct-size comparisons, competing reduct
algorithms, UCI or any real data, and any newly proposed algorithm.

## 1. Frozen populations (locked_spec_v2.md §8)

Support `x0 = (0,0,0)`, `x1 = (0,0,1)`, `x2 = (1,0,0)`, `x3 = (1,1,1)` over `C = {A,B,C}`;
masses `1/4`, `1/2−ε`, `1/4`, `ε` for `0 < ε ≤ 1/4`.

| θ | labels | population reduct family | core |
|---|---|---|---|
| P0 | (0,0,1,0) | `{{A,B}, {A,C}}` | `{A}` |
| P1 | (0,0,1,1) | `{{A}}` | `{A}` |
| P2 | (0,1,0,1) | `{{C}}` | `{C}` |

`rough_exact.py` recomputes this table at import from the labels alone; it is never typed in.

## 2. Exact empirical computation

For a labeled sample, every one of the 8 subsets of `{A,B,C}` is tested for empirical
sufficiency (no two observations agreeing on the subset carry different labels). The
inclusion-minimal sufficient sets form the complete empirical reduct family; their intersection
is the empirical core. No heuristic search, no greedy construction, duplicates allowed,
operating on labeled observations.

Two implementations are used and cross-checked in `validation.py`:

* **literal** — `empirical_family_core(obs)` consumes the labeled observations directly;
* **lookup** — the same function tabulated over the 16 possible sets of observed atoms.

They agree because labels are deterministic (§1 A2 realizability), so a sample influences the
structures only through the set of distinct labeled observations it contains. The Monte Carlo
engine uses the lookup path, which is why a repetition at `n = 50000` costs the same as at
`n = 10`: one multinomial count vector, not `n` draws.

### Derived minimal witness sets

Enumerated by `rough_exact.witness()`, not assumed — the recovering observed-sets turn out to
be a principal up-set in every case, and this is verified rather than posited:

| θ | core recovered iff seen | family recovered iff seen |
|---|---|---|
| P0 | `x0, x2` | `x0, x2, x3` |
| P1 | `x0, x2` | `x0, x2` |
| P2 | `x0, x1` | `x0, x1` |

This single table is the mechanism behind the whole stage: only the **family** target of
**P0** requires the rare atom `x3`, whose mass is ε. Recovery probabilities are then computed
as `P(all witness atoms observed)` by inclusion–exclusion, which is far better conditioned than
an alternating sum over all 16 observed sets; the general expansion is retained as a
cross-check.

## 3. Two tiers of analytic formula, never conflated

**Tier 1 — from the locked spec.** `p_core = 1 − 2(3/4)^n + (1/2)^n` for P0/P1;
`e_core^(2) = (3/4)^n + (1/2+ε)^n − (1/4+ε)^n` for P2; uniform core bound `2(3/4)^n`; exact
two-point testing risk `R*_n = ½(1−ε)^n`; rare-atom survival `(1−ε)^n`; the §12 sample sizes.

**Tier 2 — plug-in bookkeeping derived here.** Exact recovery probabilities of the plug-in
empirical estimator for both targets, from the witness table above. These are exact statements
about the frozen class, not new theory, and they are labelled `plug-in (Tier 2)` in every CSV
row that uses them.

The distinction that matters for the manuscript: `R*_n = ½(1−ε)^n` is an **estimator-independent
lower bound** on family minimax recovery over the class. The plug-in family curve is a
different object — one specific estimator — and sits above it. The experiment reports both and
never substitutes one for the other.

## 4. E0 — finite-sample recovery (`run_e0.py`)

* ε ∈ {0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001}; θ ∈ {P0, P1, P2}.
* n ∈ {1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000}, extended through
  {10000, 20000, 50000} while the exact family curve has not yet reached 0.99.
* 10 000 repetitions by default; **100 000** at the difficult points, defined mechanically as,
  for each (ε, θ, target), the grid `n` nearest the 50 % crossing of the exact curve — the
  steep region, where Monte Carlo noise is largest and an implementation error would show.
* Recorded per row: ε, n, θ, target, repetitions, successes, `p_hat`, standard error, 95 %
  Wilson interval, analytic reference and its tier, absolute error, tolerance, z-score,
  pass flag, hard-point flag, seed, runtime.
* Seeds: `numpy.random.default_rng(SeedSequence([20260912, θ, i_ε, i_n, reps]))`. The literal
  key is stored in the `seed` column, so any single point can be re-run in isolation.
* `results/e0_results.csv` is written **before** any figure is drawn, and the figures are then
  produced by reading that file back.

## 5. E0.5 — rare-event mechanism (`run_e05.py`)

First rare-atom time `T_rare = inf{n : X_n = x3}`, 10 000 repetitions per ε, sampled from the
4-way categorical law by inverse CDF — not from a Bernoulli short-cut on "is it x3" — so the
survival law is measured rather than assumed. Repetitions are advanced in blocks and only while
they have not yet seen `x3`, with a cap of `⌈25/ε⌉` draws; the censored count is recorded and
is expected to be 0 (survival at the cap is `e^{-25} ≈ 1.4·10^{-11}`).

Besides the survival curve the run records the structural state immediately **before** and
**after** the first `x3` under P0. This exists to block an overstatement: `x3` is necessary for
exact family recovery under P0, but not sufficient — if `x0` or `x2` is still unseen at that
moment, the empirical family is still wrong. The CSV reports how often that occurs.

## 6. E1 — sample-complexity separation (`run_e1.py`)

Deterministic, no simulation. For δ ∈ {0.10, 0.05, 0.01} and ε on a log grid from 0.25 down to
1e−5 (including every E0 value and {0.1, 0.01, 0.001, 0.0001}):

```
n_core^UB   = ceil( log(2/δ) / −log(3/4) )
n_family^LB = ceil( log(1/(2δ)) / −log(1−ε) )
approx      = log(1/(2δ)) / ε
Q           = n_family^LB / n_core^UB
```

The small-ε slope of `n_family^LB` is **fitted** on ε ≤ 1e−2 and reported, not forced to −1.
`ε·Q` is reported in the tail. The table at δ = 0.01 goes to `tables/e1_delta001.csv` and
`.tex` (booktabs, generated — not hand-edited).

## 7. Validation criterion

Monte Carlo agrees with an analytic reference when the absolute error is within the more
permissive of `3 × SE` and `0.01`, as the stage prompt specifies. The z-score is stored in
every row so the stricter rule stays inspectable. Every violation is flagged, listed in
`reports/validation_report.md`, and re-run with more repetitions before any failure is
declared. No point is dropped or smoothed.

If an **analytic identity** fails — a closed form disagreeing with exact enumeration — the
stage stops rather than continuing to figures.

## 8. Reproduction

```
pip install -r requirements.txt
python3 validation.py     # analytic + implementation self-test, no randomness
python3 run_e1.py         # deterministic, seconds
python3 run_e0.py         # Monte Carlo grid
python3 run_e05.py        # rare-event simulation
```

All scripts use paths relative to their own location and fixed seeds, so repeated runs
reproduce the CSVs bit-for-bit on the same numpy version.
