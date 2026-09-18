"""Exact empirical Pawlak reduction structures on the three-population construction of Section 4.1.

Everything here is enumeration over the 8 subsets of C = {a1, a2, a3}. There is no heuristic
search, no greedy reduct construction and no approximation anywhere in this file.

Definitions follow the paper's Section 3.2 and 3.3:
  * B is empirically sufficient for a labeled sample iff any two observations agreeing on B
    carry the same decision label;
  * empirical reducts are the inclusion-minimal empirically sufficient sets;
  * the empirical core is the intersection of the empirical reduct family.

Two computation paths are provided and are checked against each other in validation.py:

  literal path   `empirical_family_core(obs)` consumes the labeled observations themselves.
  lookup path    `structures_by_observed_set(theta)` tabulates the same function over the 16
                 possible sets of observed atoms. Labels are deterministic (Assumption 2,
                 realisability) and sufficiency depends on the sample only through the SET of
                 distinct labeled observations, so the two paths agree by construction -- and
                 are verified to.

The lookup path is what makes 10^5-repetition Monte Carlo at n = 20000 cheap: a repetition is
summarised by which atoms it saw, drawn as one multinomial vector.
"""
from itertools import combinations

ATTRS = ("a1", "a2", "a3")
P = 3

# Section 4.1 support and masses. Do not alter.
X0, X1, X2, X3 = (0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 1, 1)
SUPPORT = (X0, X1, X2, X3)

# theta -> labels of (x0, x1, x2, x3)
LABELS = {0: (0, 0, 1, 0), 1: (0, 0, 1, 1), 2: (0, 1, 0, 1)}
THETAS = (0, 1, 2)

ALL_SUBSETS = [frozenset(c) for r in range(P + 1) for c in combinations(range(P), r)]


def masses(eps):
    """Masses of (x0, x1, x2, x3) for 0 < eps <= 1/4, as in Section 4.1."""
    if not (0 < eps <= 0.25):
        raise ValueError(f"eps must lie in (0, 1/4], got {eps}")
    return (0.25, 0.5 - eps, 0.25, eps)


def show(B):
    return "{" + ",".join(ATTRS[a] for a in sorted(B)) + "}"


def show_family(fam):
    return "{" + ", ".join(show(B) for B in sorted(fam, key=lambda B: (len(B), sorted(B)))) + "}"


def is_sufficient(obs, B):
    """B sufficient for the labeled sample obs = [(atom, label), ...]: no two observations
    agree on B and disagree on the label."""
    seen = {}
    idx = sorted(B)
    for x, y in obs:
        k = tuple(x[a] for a in idx)
        if k in seen:
            if seen[k] != y:
                return False
        else:
            seen[k] = y
    return True


def empirical_family_core(obs):
    """Literal path: complete empirical reduct family and empirical core of a labeled sample.

    Duplicated observations are allowed and are simply redundant. An empty or single-label
    sample makes the empty set sufficient, so the family is {{}} and the core is {} -- a
    legitimate non-recovery, not an error."""
    suf = [B for B in ALL_SUBSETS if is_sufficient(obs, B)]
    fam = frozenset(B for B in suf if not any(D < B for D in suf))
    core = frozenset.intersection(*fam) if fam else frozenset()
    return fam, core


def labeled_sample(atom_indices, theta):
    """Turn indices of observed atoms into labeled observations."""
    lab = LABELS[theta]
    return [(SUPPORT[i], lab[i]) for i in atom_indices]


def observed_sets():
    """All 16 possible sets of observed atoms, as frozensets of atom indices."""
    return [frozenset(c) for r in range(len(SUPPORT) + 1)
            for c in combinations(range(len(SUPPORT)), r)]


def structures_by_observed_set(theta):
    """Lookup path: observed atom set -> (empirical family, empirical core)."""
    return {T: empirical_family_core(labeled_sample(sorted(T), theta)) for T in observed_sets()}


# population structures: the whole support is observed
POP = {}
for _t in THETAS:
    POP[_t] = empirical_family_core(labeled_sample(range(len(SUPPORT)), _t))
POP_FAMILY = {t: POP[t][0] for t in THETAS}
POP_CORE = {t: POP[t][1] for t in THETAS}

_TABLE = {t: structures_by_observed_set(t) for t in THETAS}


def recovery(theta, observed):
    """(core recovered?, family recovered?) for a set of observed atom indices."""
    fam, core = _TABLE[theta][frozenset(observed)]
    return core == POP_CORE[theta], fam == POP_FAMILY[theta]


def recovering_sets(theta, target):
    """The observed-atom sets on which the plug-in estimator recovers `target`."""
    k = 0 if target == "core" else 1
    return [T for T in observed_sets() if recovery(theta, T)[k]]


def witness(theta, target):
    """If the recovering sets form a principal up-set {T : Req subset of T}, return Req.

    This is not assumed: it is derived by enumeration and returns None when it fails, in which
    case the caller must fall back to the general observed-set expansion. It matters because
    P(Req all observed) is far better conditioned numerically than an alternating sum over all
    16 observed sets."""
    rec = recovering_sets(theta, target)
    if not rec:
        return None
    req = frozenset.intersection(*[frozenset(T) for T in rec])
    if all((req <= T) == (T in rec) for T in observed_sets()):
        return req
    return None
