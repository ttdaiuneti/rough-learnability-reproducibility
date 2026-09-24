# Exact checks of the theoretical results (Section 4 and Table 5)

Finite, exact computations that test the statements of Section 4 on enumerated domains. They
are verification, not proof: every result in the paper is proved, and no proof appeals to
these checks. Each script recomputes reduct families and cores from the literal definition of
sufficiency, independently of the characterisation it tests, and uses exact rational
arithmetic wherever probabilities occur.

## Scripts, outputs and the claims they check

| Script | Output | Row of Table 5 | Claims checked |
|---|---|---|---|
| `v2_gate_checker.py` | `gate_report.txt` | Construction | the three-population construction of Section 4.1: structures (Table 1), the exact core recovery probabilities of Proposition 3, the uniform bound of Corollary 4, the exact testing risk of Proposition 7, the sample-complexity values of Theorem 9 and Table 3, and the leave-one-out characterisation by the Bayes risk; also the geometric translation of Proposition 10 on the domains of the next row |
| `hypergraph_checker.py` | `atom_addition_report.txt` | Atom additions | Proposition 10 over all atom additions on three binary attributes (34 752) and on four binary attributes with at most four atoms (828 800); discernibility representation (Lemma 1); full-support uniqueness (Proposition 11) |
| `final_checker.py` | `geometric_translation_report.txt` | Atom additions | the geometric translation of Proposition 10 on a larger domain: all structures on three binary attributes and on four binary attributes with at most five atoms |
| `k1_k3_checker.py` | `k1_k3_results.json` | Plug-in bounds | Propositions 12 and 13 (exact plug-in errors against the bounds, deletion identities, core estimator), and the counts of supports admitting a core-preserving cover reported in Section 4.7 (348/2 820, 7 872/19 024, 1 260/6 204) |
| `brute_crosscheck_extra.py` | `brute_crosscheck_extra.json` | Plug-in bounds | brute-force cross-check of the exact error formula used by `k1_k3_checker.py` |
| `f4pp_check.py` | `f4pp_results.json` | Single-atom changes | Proposition 16 |
| `check_family_upper.py` | `check_family_upper.json` | Family upper bound | the family estimator of Theorem 14 (6 484 supports, 12 968 mass assignments, 334 separating classes, 4 008 risk evaluations), deletion radii and the multiclass core formula |
| `check_floor_class.py` | `check_floor_class.json` | Structural-zero classes | Theorem 17 and Corollary 18, and the error formulas of the Theorem 14 estimator |
| `check_thm17_constant.py` | `check_thm17_constant.json` | Structural-zero classes | the constants $K_{\mathrm{CORE}} = 2p$ and $K_{\mathcal{R}} = 2\binom{p}{\lfloor p/2\rfloor}$ of Theorem 17(b): exact plug-in errors of the completed estimator against $K_\Psi (1-\nu_\Psi)^n$ (11 391 checks) |
| `check_revision_claims.py` | `check_revision_claims.json` | Supplementary sub-claims | the sub-claims listed in the last row of Table 5 |
| `k4_e2_check.py` | `k4_results.json` | (Section 5.2) | on the seven accepted finite populations: every clutter edge of the six unique-reduct populations is a singleton, `tic-tac-toe` has an empty core, and its family radius $\mu_{fam} = 2/479$ with the comparison at $n = 1000$ quoted in Section 4.7 |

## Run

```
pip install -r requirements.txt
python3 v2_gate_checker.py
python3 hypergraph_checker.py
python3 final_checker.py
python3 k1_k3_checker.py
python3 brute_crosscheck_extra.py
python3 f4pp_check.py
python3 k4_e2_check.py
python3 check_family_upper.py
python3 check_floor_class.py
python3 check_revision_claims.py
python3 check_thm17_constant.py
```

Each script writes its output next to itself. The total run time is about ten minutes on a
laptop; `k1_k3_checker.py` takes the longest, at about 95 seconds. Random choices, such as mass
vectors and sampled supports, use fixed seeds, so every output is reproduced exactly. The only
fields that change from run to run are wall-clock times.

`k4_e2_check.py` reads the processed populations in `../empirical_populations/data/processed/`
and imports `../empirical_populations/exact_rough.py`. It needs `numpy` and `networkx`. All
other scripts use the Python standard library only.

Some script docstrings and report headers refer to the internal names of the development stages
at which the checks were written, for example "v2 gate", "K1-K3" and "D5". These are labels
only. The table above gives the correspondence with the paper.

`SHA256SUMS` lists the checksums of the outputs as distributed.
