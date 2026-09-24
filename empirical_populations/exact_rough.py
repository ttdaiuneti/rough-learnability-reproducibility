"""Exact Pawlak reduction structures for finite empirical populations (Section 5.2 of the paper).

Everything is exact: no heuristic reduct search, no top-k, no random subset search, no
single-reduct surrogate. Attribute subsets are bitmasks over `p <= 15` bits, so the
minimal-transversal step is a complete sweep of at most 32 768 subsets.

Objects (paper Sections 3.2 and 4.7):
  support        distinct labeled condition vectors of the processed dataset, with masses
                 P_N(x, y) = (row multiplicity) / N
  edges          D(u, v) = {a : u_a != v_a} for bichromatic pairs, as bitmasks
  clutter        the inclusion-minimal edges
  R              the minimal transversals of the clutter
  CORE           the intersection of R, equivalently the singleton edges of the clutter

Two independent routes to the same answers are provided and are cross-checked in validation.py:

  direct route    `family_core_from_rows` rebuilds sufficiency from the labeled observations, by
                  the definition (a subset is sufficient iff no two observations agree on it and
                  disagree on the label). Slow, obviously correct.
  clutter route   `family_core_from_edges` goes through the discernibility clutter and minimal
                  transversals. Fast, and what the experiment uses.

For the sampling experiment there is a third, much faster characterisation, which is DERIVED
here and VERIFIED against the direct route rather than assumed (see `recovery_targets`):

    the observed edge set E_obs is always a subset of the population edge set, therefore
      core recovered   <=>  every singleton edge of the population clutter is realised
      family recovered <=>  every edge of the population clutter is realised

  ("realised" = some bichromatic pair generating that edge has both atoms observed.)

  Sketch: R is determined by the clutter, so family recovery <=> clutter_emp = clutter_pop. If
  every population clutter edge is in E_obs, then since every element of E_obs contains some
  population clutter edge, min(E_obs) = clutter_pop. Conversely a missing clutter edge c cannot
  be in min(E_obs) at all, so the two clutters differ. For the core, a singleton edge is
  minimal wherever it appears, so CORE_emp = {a : {a} in E_obs}.
"""
import itertools

import numpy as np


# --------------------------------------------------------------------------- population
class Population:
    """A finite empirical population P_N built from processed rows.

    Attributes
    ----------
    X, y     support: distinct condition vectors and their labels
    counts   row multiplicities; masses = counts / N
    p, N     number of condition attributes, number of rows
    """

    def __init__(self, X, y, p=None):
        X = np.asarray(X, dtype=np.int64)
        y = np.asarray(y, dtype=np.int64)
        self.p = int(X.shape[1] if p is None else p)
        self.N = int(X.shape[0])
        keys, inv, counts = np.unique(np.column_stack([X, y]), axis=0,
                                      return_inverse=True, return_counts=True)
        self.X = keys[:, :-1]
        self.y = keys[:, -1]
        self.counts = counts
        self.masses = counts / counts.sum()
        self._edges = None
        self._clutter = None

    # ---- realisability (Assumption 2) -------------------------------------
    def realisability_conflicts(self):
        """Identical full condition vectors carrying different labels.

        Returns (n_conflicting_vectors, n_rows_involved, examples)."""
        uniq, inv = np.unique(self.X, axis=0, return_inverse=True)
        bad, rows, ex = 0, 0, []
        for k in range(len(uniq)):
            sel = inv == k
            labels = self.y[sel]
            if len(np.unique(labels)) > 1:
                bad += 1
                rows += int(self.counts[sel].sum())
                if len(ex) < 3:
                    ex.append((uniq[k].tolist(), sorted(set(labels.tolist()))))
        return bad, rows, ex

    # ---- discernibility ---------------------------------------------------
    def pair_edges(self):
        """(edge bitmask, i, j) for every bichromatic pair of support atoms."""
        i, j = np.triu_indices(len(self.X), k=1)
        bichrome = self.y[i] != self.y[j]
        i, j = i[bichrome], j[bichrome]
        diff = self.X[i] != self.X[j]
        weights = (1 << np.arange(self.p, dtype=np.int64))
        masks = (diff * weights).sum(axis=1)
        return masks, i, j

    @property
    def edges(self):
        if self._edges is None:
            self._edges = np.unique(self.pair_edges()[0])
        return self._edges

    @property
    def clutter(self):
        if self._clutter is None:
            self._clutter = minimal_masks(self.edges)
        return self._clutter

    def n_bichromatic_pairs(self):
        i, j = np.triu_indices(len(self.X), k=1)
        return int((self.y[i] != self.y[j]).sum())


# --------------------------------------------------------------------------- bitmask helpers
def minimal_masks(masks):
    """Inclusion-minimal elements of a set of bitmasks."""
    ms = np.unique(np.asarray(masks, dtype=np.int64))
    if (ms == 0).any():
        # a zero edge means two identical condition vectors carry different labels, i.e. a
        # realisability (Assumption 2) violation. Dropping it silently would hide an
        # inconsistent table behind a clean clutter, so refuse instead: the realisability
        # screen must have run first.
        raise ValueError("zero discernibility edge: the table violates realisability (Assumption 2)")
    keep = []
    for m in sorted(ms.tolist(), key=lambda v: bin(v).count("1")):
        if not any((k & m) == k for k in keep):
            keep.append(m)
    return np.array(sorted(keep), dtype=np.int64)


def minimal_transversals(clutter, p):
    """All inclusion-minimal attribute subsets meeting every edge, by complete enumeration."""
    if len(clutter) == 0:
        return np.array([0], dtype=np.int64)  # empty set is the unique minimal sufficient set
    size = 1 << p
    hits = np.ones(size, dtype=bool)
    all_masks = np.arange(size, dtype=np.int64)
    for e in np.asarray(clutter, dtype=np.int64):
        hits &= (all_masks & e) != 0
    minimal = hits.copy()
    for a in range(p):
        bit = 1 << a
        has = (all_masks & bit) != 0
        minimal[has] &= ~hits[all_masks[has] ^ bit]
    return all_masks[minimal]


def core_mask(family):
    fam = np.asarray(family, dtype=np.int64)
    if len(fam) == 0:
        return 0
    acc = fam[0]
    for f in fam[1:]:
        acc &= f
    return int(acc)


def mask_to_set(mask, p):
    return frozenset(a for a in range(p) if mask >> a & 1)


def singleton_edges(masks):
    return np.array(sorted(int(m) for m in np.asarray(masks) if m and (m & (m - 1)) == 0),
                    dtype=np.int64)


# --------------------------------------------------------------------------- the two routes
def family_core_from_rows(X, y, p):
    """DIRECT route: sufficiency straight from the definition, over all 2^p subsets."""
    X = np.asarray(X, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    suff = []
    for mask in range(1 << p):
        cols = [a for a in range(p) if mask >> a & 1]
        if len(X) == 0:
            suff.append(mask)
            continue
        proj = X[:, cols] if cols else np.zeros((len(X), 1), dtype=np.int64)
        keys, inv = np.unique(proj, axis=0, return_inverse=True)
        ok = True
        for k in range(len(keys)):
            if len(np.unique(y[inv == k])) > 1:
                ok = False
                break
        if ok:
            suff.append(mask)
    fam = [m for m in suff if not any((s & m) == s and s != m for s in suff)]
    fam = np.array(sorted(fam), dtype=np.int64)
    return fam, core_mask(fam)


def family_core_from_edges(edge_masks, p):
    """CLUTTER route: minimal transversals of the clutter of the observed edge set."""
    clutter = minimal_masks(edge_masks) if len(edge_masks) else np.array([], dtype=np.int64)
    fam = minimal_transversals(clutter, p)
    return fam, core_mask(fam)


# --------------------------------------------------------------------------- recovery targets
def recovery_targets(pop):
    """The edge sets whose realisation is equivalent to exact recovery.

    Returns (core_target, family_target) as bitmask arrays: the singleton edges of the
    population clutter, and the whole population clutter."""
    clutter = pop.clutter
    return singleton_edges(clutter), np.asarray(clutter, dtype=np.int64)


def edge_pair_index(pop, targets):
    """For each target edge, the support-atom pairs that generate it, ordered by descending
    joint mass so that the sampling check can short-circuit on the likely pairs first."""
    masks, i, j = pop.pair_edges()
    out = {}
    for e in np.asarray(targets, dtype=np.int64):
        sel = masks == e
        ii, jj = i[sel], j[sel]
        order = np.argsort(-(pop.masses[ii] * pop.masses[jj]))
        out[int(e)] = (ii[order], jj[order])
    return out


def realised(observed, pairs):
    """Vectorised over repetitions: is this edge realised by the observed atoms?

    `observed` is a (reps, n_atoms) boolean matrix; `pairs` is (I, J). Pairs are scanned in
    the given (mass-descending) order with an active set of not-yet-realised repetitions, which
    is exact and usually terminates after a handful of pairs."""
    I, J = pairs
    out = np.zeros(observed.shape[0], dtype=bool)
    active = np.arange(observed.shape[0])
    step = 64
    for s in range(0, len(I), step):
        if active.size == 0:
            break
        ii, jj = I[s:s + step], J[s:s + step]
        hit = (observed[np.ix_(active, ii)] & observed[np.ix_(active, jj)]).any(axis=1)
        out[active[hit]] = True
        active = active[~hit]
    return out


def population_summary(pop):
    fam = minimal_transversals(pop.clutter, pop.p)
    core = core_mask(fam)
    sizes = [bin(int(m)).count("1") for m in fam]
    return dict(
        support=len(pop.X), n_edges=len(pop.edges), n_clutter=len(pop.clutter),
        n_reducts=len(fam), min_reduct=min(sizes) if sizes else 0,
        median_reduct=float(np.median(sizes)) if sizes else 0.0,
        max_reduct=max(sizes) if sizes else 0,
        core_size=bin(core).count("1"), core_mask=int(core),
        family=[int(m) for m in fam])


def show_mask(mask, p, names=None):
    names = names or [f"a{i}" for i in range(p)]
    return "{" + ",".join(names[a] for a in range(p) if mask >> a & 1) + "}"


def all_subsets_of(mask, p):
    bits = [a for a in range(p) if mask >> a & 1]
    for r in range(len(bits) + 1):
        for c in itertools.combinations(bits, r):
            yield sum(1 << a for a in c)
