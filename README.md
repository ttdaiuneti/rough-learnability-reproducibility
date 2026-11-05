# Reproducibility package

Code and data for the computational results reported in:

> Thanh Dai Tran. *Recovering the Reduct Family and the Core from Random Samples: Minimax Rates
> and an Unbounded Separation.*

This package contains the code and data needed to reproduce every number in Section 7 and Appendix B of the
paper, and the exact finite checks of the theoretical results that are summarised in Table 5
(Appendix A) and quoted in Sections 4–6. It does not include the figure- or table-formatting code
used to typeset the manuscript.

## Contents

- **`theory_checks/`** covers Sections 4–6 and Table 5. It contains exhaustive and exact-rational
  checks of the statements of Sections 4–6 on enumerated finite domains. Its README maps each
  script to the row of Table 5 and the results it checks.
- **`synthetic_validation/`** covers Section 7.1 and Appendix B. It contains exact and Monte Carlo checks
  on the three-population construction of Section 4.1, the deterministic sample-complexity
  bounds of Theorem 9, and the experiment protocol.
- **`empirical_populations/`** covers Section 7.2. It contains the screening, the exact
  structure computation, and the sampling-with-replacement recovery curves on finite empirical
  populations built from public UCI datasets. It also contains the screening
  plan, the preprocessing specification and the computational budget, together with their
  checksums. These three documents were fixed before the data were downloaded (internally, not in a public registry).

Each subdirectory has its own `README.md` with the exact commands to reproduce its results,
and its own `requirements.txt`. Checksums of the distributed outputs are in the `SHA256SUMS`
files and in `empirical_populations/data/hashes/`.

## Scope

Every number quoted in Section 7, and every count quoted in Sections 4–6 from an enumeration, is
reproducible from what is here. The theoretical results of Sections 3–6 do not depend on
this code. The checks are independent numerical verification of the stated results, not a
substitute for the proofs.
