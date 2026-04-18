"""
Orbital-level residual diagnostic.

Runs 20 SCF iterations with v3 (N=1, fast) and logs full gamma_mixed at each
step, then analyses which orbitals consistently drive max|Δγ| and whether they
fall in fragment-exclusive regions or fragment-overlap regions.

Output: outs_orbital_diagnostic/
"""
import sys, os, json, datetime
import numpy as np

PROJECT_ROOT = os.path.expanduser('~/Proj_Flow')
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import trimci
from TrimCI_Flow.fragment import (
    fragment_by_sliding_window, extract_fragment_integrals, fragment_electron_count
)
from TrimCI_Flow.trimci_adapter import solve_fragment_trimci
from TrimCI_Flow.trimci_flow import compute_fragment_rdm1, dress_integrals_meanfield
from TrimCI_Flow.trimci_flow_v2 import _assemble_global_rdm1_diag_v2

OUTPUT_DIR = 'outs_orbital_diagnostic'
os.makedirs(OUTPUT_DIR, exist_ok=True)

FCIDUMP = (
    'Fe4S4_251230orbital_-327.1920_10kdets/'
    'Fe4S4_251230orbital_-327.1920_10kdets/'
    'fcidump_cycle_6'
)
TRIMCI_CONFIG = {
    'threshold': 0.03, 'max_final_dets': 200, 'max_rounds': 2,
    'num_runs': 1, 'pool_build_strategy': 'heat_bath', 'verbose': False,
}
N_ITERS   = 20
N_AVG     = 5     # use N=5 so results match the best-known v3 run
DAMPING   = 0.1

# ---- Setup ----
h1, eri, n_elec, n_orb, E_nuc, n_alpha, n_beta, psym = trimci.read_fcidump(FCIDUMP)
order = np.argsort(np.diag(h1))

_dets_path = os.path.join(os.path.dirname(os.path.abspath(FCIDUMP)), "dets.npz")
if os.path.exists(_dets_path):
    _ref_data      = np.load(_dets_path)
    ref_alpha_bits = int(_ref_data["dets"][0, 0])
    ref_beta_bits  = int(_ref_data["dets"][0, 1])
else:
    ref_alpha_bits = int(sum(1 << int(order[i]) for i in range(n_alpha)))
    ref_beta_bits  = int(sum(1 << int(order[i]) for i in range(n_beta)))

fragments_all = fragment_by_sliding_window(n_orb, order, 15, 10)
valid = []
for frag_orbs in fragments_all:
    na, nb = fragment_electron_count(ref_alpha_bits, ref_beta_bits, frag_orbs)
    n_frag = len(frag_orbs)
    if na == 0 or nb == 0 or na > n_frag or nb > n_frag:
        continue
    h1_f, eri_f = extract_fragment_integrals(h1, eri, frag_orbs)
    valid.append((frag_orbs, na, nb, h1_f, eri_f))

n_elec_total = n_alpha + n_beta

# Orbital assignment map
frag_sets   = [set(f[0]) for f in valid]
n_frag_total = len(valid)

# Each orbital: which fragments does it belong to?
orb_fragments = {r: [] for r in range(n_orb)}
for fi, (frag_orbs, *_) in enumerate(valid):
    for r in frag_orbs:
        orb_fragments[r].append(fi)

# Category per orbital
def orb_category(r):
    frags = orb_fragments[r]
    if len(frags) == 0: return 'uncovered'
    if len(frags) == 1: return f'excl_F{frags[0]}'
    return '+'.join(f'F{f}' for f in sorted(frags))

orb_cats = {r: orb_category(r) for r in range(n_orb)}
print("Orbital categories:")
from collections import Counter
cat_counts = Counter(orb_cats.values())
for k, v in sorted(cat_counts.items()):
    print(f"  {k}: {v} orbitals -> {sorted(r for r in range(n_orb) if orb_cats[r]==k)}")

# ---- SCF loop with per-orbital logging ----
gamma_ref_vec = np.array(
    [((ref_alpha_bits >> r) & 1) + ((ref_beta_bits >> r) & 1) for r in range(n_orb)],
    dtype=np.float64)
gamma_mixed = gamma_ref_vec.copy()

gamma_history = [gamma_mixed.copy()]   # gamma_mixed at start of each iteration
delta_history = []                     # per-orbital |Δγ| at each iteration

print(f"\nStarting {N_ITERS}-iteration orbital diagnostic at {datetime.datetime.now().isoformat()}")
for it in range(N_ITERS):
    frag_rdms = []
    for (frag_orbs, na, nb, h1_bare, eri_f) in valid:
        ext_orbs  = [r for r in range(n_orb) if r not in set(frag_orbs)]
        ext_gamma = gamma_mixed[np.asarray(ext_orbs, dtype=np.intp)]
        h1_use    = dress_integrals_meanfield(h1_bare, eri, frag_orbs, ext_gamma, ext_orbs)
        if N_AVG <= 1:
            res = solve_fragment_trimci(h1_use, eri_f, na, nb, len(frag_orbs), TRIMCI_CONFIG)
            frag_rdms.append(compute_fragment_rdm1(res.dets, res.coeffs, res.n_orb_frag))
        else:
            gamma_sum = None
            for _ in range(N_AVG):
                res = solve_fragment_trimci(h1_use, eri_f, na, nb, len(frag_orbs), TRIMCI_CONFIG)
                g   = compute_fragment_rdm1(res.dets, res.coeffs, res.n_orb_frag)
                gamma_sum = g.copy() if gamma_sum is None else gamma_sum + g
            frag_rdms.append(gamma_sum / N_AVG)

    fragment_orbs_list = [f[0] for f in valid]
    gamma_new = _assemble_global_rdm1_diag_v2(
        frag_rdms, fragment_orbs_list, n_orb, n_elec_total, ref_alpha_bits, ref_beta_bits)
    gamma_mixed_prev = gamma_mixed.copy()
    gamma_mixed = DAMPING * gamma_new + (1.0 - DAMPING) * gamma_mixed
    gamma_history.append(gamma_mixed.copy())

    per_orb_delta = np.abs(gamma_mixed - gamma_mixed_prev)
    delta_history.append(per_orb_delta)
    top3_orbs = np.argsort(per_orb_delta)[-3:][::-1]
    top3_cats = [orb_cats[r] for r in top3_orbs]
    print(f"  Iter {it:2d}: max|Δγ|={per_orb_delta.max():.4f}  "
          f"top3 orbs={list(top3_orbs)} cats={top3_cats}")

print(f"Done at {datetime.datetime.now().isoformat()}")

# ---- Analysis ----
delta_array = np.stack(delta_history, axis=0)  # (N_ITERS, n_orb)

# Mean per-orbital delta
mean_delta = delta_array.mean(axis=0)
std_delta  = delta_array.std(axis=0)

# Frequency of appearing in top-3
top3_counts = np.zeros(n_orb, dtype=int)
for row in delta_array:
    for r in np.argsort(row)[-3:][::-1]:
        top3_counts[r] += 1

# Per-category mean delta
cat_deltas = {}
for r in range(n_orb):
    cat = orb_cats[r]
    if cat not in cat_deltas:
        cat_deltas[cat] = []
    cat_deltas[cat].append(mean_delta[r])

print("\n=== Per-category mean |Δγ| ===")
for cat, vals in sorted(cat_deltas.items()):
    arr = np.array(vals)
    print(f"  {cat}: n={len(arr)}  mean={arr.mean():.5f}  max={arr.max():.5f}  "
          f"min={arr.min():.5f}  std={arr.std():.5f}")

print("\n=== Top 10 orbitals by mean |Δγ| ===")
top10_orbs = np.argsort(mean_delta)[-10:][::-1]
for r in top10_orbs:
    print(f"  orb {r:2d}: mean_delta={mean_delta[r]:.5f}  std={std_delta[r]:.5f}  "
          f"top3_count={top3_counts[r]:2d}/{N_ITERS}  cat={orb_cats[r]}")

print("\n=== Top 10 orbitals by top-3 frequency ===")
top10_freq = np.argsort(top3_counts)[-10:][::-1]
for r in top10_freq:
    if top3_counts[r] > 0:
        print(f"  orb {r:2d}: top3_count={top3_counts[r]:2d}/{N_ITERS}  "
              f"mean_delta={mean_delta[r]:.5f}  cat={orb_cats[r]}")

# Is max|Δγ| orbital consistent across iterations?
max_orb_per_iter = [int(np.argmax(row)) for row in delta_array]
from collections import Counter
max_orb_freq = Counter(max_orb_per_iter)
print(f"\n=== Max-Δγ orbital frequency (which orbital is max most often) ===")
for orb, cnt in max_orb_freq.most_common(6):
    print(f"  orb {orb:2d}: {cnt}/{N_ITERS} iters  cat={orb_cats[orb]}")

# Save
results = {
    'n_iters':         N_ITERS,
    'n_gamma_avg':     N_AVG,
    'fragment_orbs':   [list(f[0]) for f in valid],
    'orb_categories':  {str(r): orb_cats[r] for r in range(n_orb)},
    'mean_delta':      mean_delta.tolist(),
    'std_delta':       std_delta.tolist(),
    'top3_counts':     top3_counts.tolist(),
    'max_orb_per_iter': max_orb_per_iter,
    'per_category_mean': {cat: float(np.mean(vals)) for cat, vals in cat_deltas.items()},
    'per_category_max':  {cat: float(np.max(vals))  for cat, vals in cat_deltas.items()},
    'delta_array':     delta_array.tolist(),  # full per-orbital, per-iter deltas
}
with open(f'{OUTPUT_DIR}/orbital_diagnostic.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {OUTPUT_DIR}/orbital_diagnostic.json")
