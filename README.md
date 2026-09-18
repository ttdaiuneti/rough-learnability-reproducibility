# Reproducibility package

Code and data for the experimental results reported in:

> Thanh Dai Tran. *Statistical Identification of Rough-Set Structures: Core-Family Separation
> and Witness-Cover Rates.*

This package contains exactly the code and data needed to reproduce the numbers reported in
Section 5 and Appendix A of the paper: it does not include the figure- or table-formatting
code used to typeset the manuscript.

## Contents

- **`synthetic_validation/`** -- Sections 5.2-5.4. Exact and Monte Carlo checks on the
  three-population construction of Section 4.1, and the deterministic sample-complexity
  bounds of Theorem 9.
- **`empirical_populations/`** -- Section 5.5. Screening, exact structure computation, and
  sampling-with-replacement recovery curves on finite empirical populations built from public
  UCI datasets.

Each subdirectory has its own `README.md` with the exact commands to reproduce its results,
and its own `requirements.txt`.

## Scope

Every number quoted in Sections 5.2-5.5 and the computational-checks table of Appendix A of
the paper is reproducible from what is here. The theoretical results (Sections 3-4) do not
depend on this code; the checks here are independent numerical verification of the closed
forms, not a substitute for the proofs.
