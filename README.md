# Reproducibility package

Code and data for the computational results reported in:

> Thanh Dai Tran. *Statistical Identification of Rough-Set Structures: Core-Family Separation
> and Witness-Cover Rates.*

This package contains the code and data needed to reproduce every number in Section 5 of the
paper, and the exact finite checks of the theoretical results that are summarised in Table 5
(Appendix A) and quoted in Section 4. It does not include the figure- or table-formatting code
used to typeset the manuscript.

## Contents

- **`theory_checks/`** covers Section 4 and Table 5. It contains exhaustive and exact-rational
  checks of the statements of Section 4 on enumerated finite domains. Its README maps each
  script to the row of Table 5 and the results it checks.
- **`synthetic_validation/`** covers Sections 5.2-5.4. It contains exact and Monte Carlo checks
  on the three-population construction of Section 4.1, the deterministic sample-complexity
  bounds of Theorem 9, and the experiment protocol.
- **`empirical_populations/`** covers Section 5.5. It contains the screening, the exact
  structure computation, and the sampling-with-replacement recovery curves on finite empirical
  populations built from public UCI datasets. It also contains the pre-registered screening
  plan, the preprocessing specification and the computational budget, together with their
  checksums.

Each subdirectory has its own `README.md` with the exact commands to reproduce its results,
and its own `requirements.txt`. Checksums of the distributed outputs are in the `SHA256SUMS`
files and in `empirical_populations/data/hashes/`.

## Scope

Every number quoted in Section 5, and every count quoted in Section 4 from an enumeration, is
reproducible from what is here. The theoretical results of Sections 3 and 4 do not depend on
this code. The checks are independent numerical verification of the stated results, not a
substitute for the proofs.
