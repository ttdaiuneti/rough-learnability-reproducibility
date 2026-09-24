"""Extra Mobius-vs-brute cross-checks for p=4 binary and p=3 multiclass (audit correction 12)."""
import json, os, random, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from k1_k3_checker import Support, supports, random_masses, exact_error, brute_error
rng = random.Random(915)
out = {}
for name, p, labels, kmax in [("p4_binary", 4, (0, 1), 4), ("p3_multiclass", 3, (0, 1, 2), 4)]:
    allsup = list(supports(p, labels, 2, kmax))
    picks = rng.sample(allsup, 150)
    checks = mism = 0
    for S, lab in picks:
        sup = Support(S, lab, p)
        masses = random_masses(rng, sup.k, rng.choice(["uniform", "rare"]))
        for n in (1, 2, 3, 4):
            if sup.k ** n > 256:
                continue
            checks += 1
            if brute_error(sup.ind_f, masses, n) != exact_error(sup.c_fam, masses, n) or \
               brute_error(sup.ind_c, masses, n) != exact_error(sup.c_core, masses, n):
                mism += 1
    out[name] = dict(supports_sampled=len(picks), checks=checks, mismatches=mism)
    print(name, out[name], flush=True)
json.dump(out, open(os.path.join(HERE, "brute_crosscheck_extra.json"), "w"), indent=1)
