# Phase B Context — TrimCI_Flow Mean-Field Coupling

**Purpose of this file:** Complete reference for implementing Phase B. Future sessions load this instead of re-exploring the codebase. Written 2026-04-13. Update after any non-trivial change.

---

## What Phase B Is

Phase C (complete) solves overlapping sliding-window fragments of Fe4S4 in isolation. Phase B adds a self-consistent mean-field loop: fragment 1-RDMs dress each fragment's one-body Hamiltonian with contributions from the other fragments' electrons, and we iterate to convergence.

**Scientific goal:** Test whether mean-field coupling reduces total determinant usage compared to the uncoupled Path C baseline (118 dets vs 10,095 brute-force).

**Phase B formula:**
```
h1_eff[p, q] = h1_frag[p, q]
             + Σ_{r ∈ external_orbs} γ_r · (2·eri_full[p_full, q_full, r, r]
                                            − eri_full[p_full, r, r, q_full])
```
- `p, q` are fragment-local indices (0..n_frag-1)
- `p_full = fragment_orbs[p]`, `q_full = fragment_orbs[q]` (full-system indices)
- `r` ranges over external_orbs (full-system indices NOT in this fragment)
- `γ_r = γ_global[r]` — spin-summed spatial occupation (0..2), averaged over all fragments that contain orbital r
- `eri_full` is the full (36,36,36,36) system ERI in chemist notation

---

## Environment

```bash
source /home/unfunnypanda/Proj_Flow/qflowenv/bin/activate
# Python 3.12.3
# TrimCI installed editable; C++ extension compiled
python -c "import trimci; from trimci.trimci_core import compute_1rdm; print('OK')"
```

---

## Reference Data

| Item | Path |
|------|------|
| FCIDUMP | `/home/unfunnypanda/Proj_Flow/Fe4S4_251230orbital_-327.1920_10kdets/Fe4S4_251230orbital_-327.1920_10kdets/fcidump_cycle_6` |
| dets.npz | same directory — `data["dets"]` shape (10095, 2) uint64 [alpha_bits, beta_bits] |
| progress.md | `/home/unfunnypanda/Proj_Flow/TrimCI_Flow/progress.md` |

`trimci.read_fcidump(path)` → `(h1(36,36), eri(36,36,36,36), n_elec=54, n_orb=36, E_nuc=0.0, n_alpha=27, n_beta=27, psym)`. ERI is chemist notation, 8-fold symmetry.

**CRITICAL:** Always load `dets.npz` row 0 as the reference determinant (not HF). HF causes fragments 0/1 to degenerate to 1 det each. E_nuc is already 0.0 — do not add it.

**Path C baseline result:** 3 fragments, `n_dets = [51, 51, 16]`, `total = 118`. Fragment energies are NOT summable.

---

## File Inventory

### `trimci_flow.py:21-29` — `FragmentedRunResult` dataclass

```python
@dataclass
class FragmentedRunResult:
    fragment_energies: list[float]
    fragment_n_dets: list[int]
    fragment_orbs: list[list[int]]
    total_dets: int
    brute_force_dets: int = 10095
    iterations: int = 1
```
Phase B will extend with 4 new fields (backward-compatible defaults):
```python
    iteration_history: list[dict] = field(default_factory=list)
    converged: bool = False
    convergence_delta: float = float('inf')
    convergence_delta_rdm: float = float('inf')
```
`iteration_history[k] = {"iteration": k, "energies": [...], "n_dets": [...], "delta_E": float, "delta_rdm": float}`

### `trimci_flow.py:36-135` — `run_fragmented_trimci` (PATH C — DO NOT MODIFY)

Key steps: L81 reads FCIDUMP → L84 sorts orbs by diag(h1) → L92-99 loads dets.npz[0] as ref det (HF fallback) → L102 creates fragments → L109-126 solves each fragment.

### `trimci_flow.py:142-237` — Phase B stubs (all `raise NotImplementedError`)

```python
def compute_fragment_rdm1(dets: list, coeffs: list, n_orb_frag: int) -> np.ndarray:
    # Returns (n_orb_frag, n_orb_frag) spin-summed 1-RDM
    raise NotImplementedError

def dress_integrals_meanfield(
    h1_frag: np.ndarray,           # (n_frag, n_frag)
    eri_full: np.ndarray,          # (n_orb, n_orb, n_orb, n_orb) full-system
    fragment_orbs: list[int],      # full-system orbital indices of this fragment
    external_rdm1_diag: np.ndarray,  # (n_external,) — parallel to external_orbs!
    external_orbs: list[int],      # full-system indices NOT in this fragment
) -> np.ndarray:                   # (n_frag, n_frag) h1_eff
    raise NotImplementedError

def run_selfconsistent_fragments(
    fcidump_path: str,
    window_size: int,
    stride: int,
    max_iterations: int = 20,
    convergence: float = 1e-6,
    trimci_config: Optional[dict] = None,
) -> FragmentedRunResult:
    raise NotImplementedError
```

### `fragment.py` — NO CHANGES for Phase B

```python
fragment_by_sliding_window(n_orb, order, window_size, stride) -> list[list[int]]   # L24
extract_fragment_integrals(h1_full, eri_full, fragment_orbs) -> (h1_frag, eri_frag) # L107
fragment_electron_count(ref_alpha_bits, ref_beta_bits, fragment_orbs) -> (na, nb)    # L140
```

### `trimci_adapter.py` — NO CHANGES for Phase B

```python
@dataclass FragmentResult:  # L27
    energy: float; n_dets: int
    dets: list       # TrimCI C++ Determinant objects — feed directly to compute_1rdm
    coeffs: list     # Python list of floats
    n_orb_frag: int; n_alpha_frag: int; n_beta_frag: int

solve_fragment_trimci(h1_frag, eri_frag, na, nb, n_orb_frag, config=None) -> FragmentResult  # L49
```
DEFAULT_CONFIG: `{threshold: 0.06, max_final_dets: "auto", max_rounds: 2, ...}`
max_final_dets="auto" → `int(3000/n_orb^1.5) ≈ 51` for n_orb=15.

### `analysis.py` — add 2 functions for Phase B, do not modify `determinant_summary`

```python
determinant_summary(result) -> dict   # L24 — already implemented, keep as-is
# Phase B additions:
iteration_summary(result) -> list[dict]    # print iteration table, return iteration_history
convergence_summary(result) -> dict        # print verdict, return convergence stats
```

---

## Upstream TrimCI API

### DO NOT REIMPLEMENT — use the C++ binding:
```python
from trimci.trimci_core import compute_1rdm, compute_1rdm_spin_resolved, energy_from_rdm

# Spin-summed 1-RDM (what Phase B needs):
gamma_flat = compute_1rdm(dets, list(coeffs), n_orb_frag)
gamma = np.asarray(gamma_flat).reshape(n_orb_frag, n_orb_frag)
# gamma[r, r] ∈ [0, 2]; trace = n_alpha + n_beta

# Spin-resolved (for validation/debug only):
gamma_aa_flat, gamma_bb_flat = compute_1rdm_spin_resolved(dets, list(coeffs), n_orb_frag)
gamma_aa = np.array(gamma_aa_flat).reshape(n_orb_frag, n_orb_frag)
gamma_bb = np.array(gamma_bb_flat).reshape(n_orb_frag, n_orb_frag)

# Energy cross-check (optional):
# E = energy_from_rdm(gamma, Gamma2, h1, eri, E_nuc, n_orb)
```
Binding source: `TrimCI-dev-main/cpp/trimci_core/bindings/bind_rdm.cpp:20-28`.
Usage pattern: `TrimCI-dev-main/py/trimci/lvcc/rdm_provider_v2/base.py:63-91`.

**WARNING:** `trimci.attentive_trimci.compute_1rdm` is NOT the right function — it returns diagonal occupation vectors only, not the full matrix.

---

## Design Decisions (locked — see plan file for full rationale)

| ID | Decision | Why |
|----|----------|-----|
| D1 | Spin-summed 1-RDM (restricted closure) | Formula requires total spatial occupation; single h1 channel |
| D2 | Average γ over overlapping fragments | W=15 S=10 gives 5-orb overlaps; averaging preserves [0,2] range |
| D3 | Ref-det fallback for uncovered orbitals | Defensive coding; in practice never triggered with 3 fragments |
| D4 | DUAL convergence: max\|ΔE\| < 1e-6 AND max\|Δγ\| < 1e-4 | Energy alone plateaus from auto-cap screening |
| D5 | Dress from bare h1_frag every iteration | Standard SCF; no drift accumulation |
| **D6** | **Linear mixing: γ_mixed = α·γ_new + (1-α)·γ_prev, default α=0.5** | **CRITICAL — CI SCF oscillates without damping** |
| D7 | Assert h1_eff symmetry (atol=1e-12) | Algebraically guaranteed; assertion catches indexing bugs |
| D8 | Freeze (na, nb) at iter 0 | Dressing must not change fragment electron count |
| D9 | Extend FragmentedRunResult (4 new optional fields) | Backward compatible; no new dataclass needed |
| D10 | external_rdm1_diag length = len(external_orbs) | Matches stub docstring; caller slices γ_global |
| D11 | Duplicate Path C setup in run_selfconsistent_fragments | No refactor of working code |
| D12 | try/finally always appends progress.md | Failure record is as important as success record |

---

## Phase B Implementation Sketches

### `compute_fragment_rdm1` (simple wrapper)
```python
def compute_fragment_rdm1(dets, coeffs, n_orb_frag):
    from trimci.trimci_core import compute_1rdm
    gamma_flat = compute_1rdm(dets, list(coeffs), n_orb_frag)
    return np.asarray(gamma_flat, dtype=np.float64).reshape(n_orb_frag, n_orb_frag)
```

### `dress_integrals_meanfield` (NumPy vectorised)
```python
def dress_integrals_meanfield(h1_frag, eri_full, fragment_orbs,
                              external_rdm1_diag, external_orbs):
    fa = np.asarray(fragment_orbs, dtype=np.intp)
    ea = np.asarray(external_orbs, dtype=np.intp)
    gamma_r = np.asarray(external_rdm1_diag, dtype=np.float64)
    # J: 2 * sum_r gamma_r * eri[p_full, q_full, r, r]
    J_block = eri_full[np.ix_(fa, fa, ea, ea)]             # (nF, nF, nE, nE)
    J_diag  = np.diagonal(J_block, axis1=2, axis2=3)       # (nF, nF, nE)
    J_term  = 2.0 * np.einsum('pqr,r->pq', J_diag, gamma_r)
    # K: sum_r gamma_r * eri[p_full, r, r, q_full]
    K_block = eri_full[np.ix_(fa, ea, ea, fa)]             # (nF, nE, nE, nF)
    K_diag  = np.diagonal(K_block, axis1=1, axis2=2)       # (nF, nF, nE)
    K_term  = np.einsum('pqr,r->pq', K_diag, gamma_r)
    h1_eff = h1_frag + J_term - K_term
    assert np.allclose(h1_eff, h1_eff.T, atol=1e-12)
    return h1_eff
```
`np.diagonal(X, axis1=i, axis2=j)` appends the diagonal axis at the END of the shape.
Shape check for J: `(nF, nF, nE, nE)` → diagonal on axes 2,3 → `(nF, nF, nE)`. ✓
Shape check for K: `(nF, nE, nE, nF)` → diagonal on axes 1,2 → `(nF, nF, nE)` where K_diag[p,q,r] = eri_full[fa[p], ea[r], ea[r], fa[q]]. ✓

### `_assemble_global_rdm1_diag` (private helper, add above run_selfconsistent_fragments)
```python
def _assemble_global_rdm1_diag(fragment_rdm1s, fragment_orbs_list, n_orb, ref_alpha_bits, ref_beta_bits):
    total = np.zeros(n_orb); count = np.zeros(n_orb, dtype=int)
    for gamma_F, frag_orbs in zip(fragment_rdm1s, fragment_orbs_list):
        for local_idx, full_idx in enumerate(frag_orbs):
            total[full_idx] += gamma_F[local_idx, local_idx]
            count[full_idx] += 1
    global_diag = np.zeros(n_orb)
    for r in range(n_orb):
        if count[r] > 0:
            global_diag[r] = total[r] / count[r]
        else:  # ref-det fallback
            global_diag[r] = ((ref_alpha_bits >> r) & 1) + ((ref_beta_bits >> r) & 1)
    return global_diag
```

### `run_selfconsistent_fragments` — iteration loop skeleton
```python
# Setup (duplicated from run_fragmented_trimci:80-117 — do NOT refactor Path C):
h1, eri, n_elec, n_orb, E_nuc, n_alpha, n_beta, psym = trimci.read_fcidump(fcidump_path)
order = np.argsort(np.diag(h1))
# ... load ref det, build fragments, filter degenerate, cache h1_bare and eri_frag ...

gamma_mixed = None
prev_energies = prev_gamma = None
iteration_history = []
converged = False; delta_E = delta_rdm = float('inf')

for it in range(max_iterations + 1):
    frag_results = []
    for (frag_orbs, na, nb, h1_bare, eri_f) in valid:
        if it == 0:
            h1_use = h1_bare
        else:
            ext_orbs = [r for r in range(n_orb) if r not in set(frag_orbs)]
            h1_use = dress_integrals_meanfield(
                h1_bare, eri, frag_orbs,
                gamma_mixed[np.asarray(ext_orbs, dtype=np.intp)], ext_orbs)
        frag_results.append(solve_fragment_trimci(h1_use, eri_f, na, nb, len(frag_orbs), trimci_config))

    # Build new γ and apply damping
    gamma_new = _assemble_global_rdm1_diag(
        [compute_fragment_rdm1(r.dets, r.coeffs, r.n_orb_frag) for r in frag_results],
        [f[0] for f in valid], n_orb, ref_alpha_bits, ref_beta_bits)
    gamma_mixed = gamma_new if gamma_mixed is None else damping * gamma_new + (1-damping) * gamma_mixed

    energies = [r.energy for r in frag_results]
    if prev_energies: delta_E = max(abs(a-b) for a,b in zip(energies, prev_energies))
    if prev_gamma: delta_rdm = float(np.max(np.abs(gamma_mixed - prev_gamma)))

    iteration_history.append({"iteration": it, "energies": energies,
                               "n_dets": [r.n_dets for r in frag_results],
                               "delta_E": delta_E, "delta_rdm": delta_rdm})
    prev_energies = energies; prev_gamma = gamma_mixed.copy(); final_results = frag_results

    if it > 0 and delta_E < convergence and delta_rdm < rdm_convergence:
        converged = True; break
```

**Note on signature extension:** Add `rdm_convergence=1e-4, damping=0.5` to `run_selfconsistent_fragments` — the current stub signature only has `convergence` for energy. These params are new but needed.

---

## Verification Commands

```bash
cd /home/unfunnypanda/Proj_Flow
source qflowenv/bin/activate

# 1. Path C regression (must still return 118 total dets)
python -c "
from TrimCI_Flow import run_fragmented_trimci, determinant_summary
r = run_fragmented_trimci(
    'Fe4S4_251230orbital_-327.1920_10kdets/Fe4S4_251230orbital_-327.1920_10kdets/fcidump_cycle_6',
    window_size=15, stride=10)
determinant_summary(r)
assert r.total_dets == 118 and r.fragment_n_dets == [51, 51, 16]
print('Path C preserved.')
"

# 2. Phase B smoke test
python -c "
from TrimCI_Flow import run_selfconsistent_fragments, iteration_summary, convergence_summary, determinant_summary
r = run_selfconsistent_fragments(
    'Fe4S4_251230orbital_-327.1920_10kdets/Fe4S4_251230orbital_-327.1920_10kdets/fcidump_cycle_6',
    window_size=15, stride=10,
    max_iterations=20, convergence=1e-6, rdm_convergence=1e-4, damping=0.5)
iteration_summary(r)
convergence_summary(r)
determinant_summary(r)
"
```

Expected Phase B outputs to check:
- Iter 0 energies match Path C within solver tolerance
- delta_E and delta_rdm are finite after iter 1
- Per-fragment energies change 3rd–6th decimal between iter 0 and final (dressing doing something)
- No NaN in γ_global; no symmetry assertion failure

---

## progress.md Append Protocol

**Always append** — never overwrite. Template structure:
```
# Phase B — Run N
## Date / Objective / Why / Expected behaviour
## Files touched / Functions added or changed (list ALL)
## Implementation details (reference D1-D12)
## Inputs used (FCIDUMP, W, S, TrimCI config, iteration controls)
## Execution results (iteration table, convergence flag, symmetry checks)
## Observed behaviour / Bugs or issues / Fixes applied
## Output interpretation (did dressing do anything? stable loop?)
## Status: SUCCESS / PARTIAL / FAILURE
## What remains unresolved / Next step
## Phase C integrity check (re-run Path C after changes; confirm 118 dets)
```

---

## Plan File

Full implementation plan (design decisions, code-by-code spec, verification steps, agent assignment):
`/home/unfunnypanda/.claude/plans/humming-singing-reddy.md`
