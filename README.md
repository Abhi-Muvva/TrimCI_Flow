# TrimCI_Flow

Fragmented active-space CI for Fe4S4, benchmarking QFlow-style classical fragmentation
with TrimCI as the sub-solver.

**Target:** reproduce brute-force TrimCI energy (−327.1920 Ha) with fewer total determinants.  
**System:** Fe4S4, 36 spatial orbitals, 27α + 27β electrons.

---

## Background

### QFlow

QFlow is a fragmentation framework for large active-space problems. The core idea is
borrowed from quantum embedding methods like DMET: split a large active space into smaller
overlapping fragments, solve each fragment independently with a sub-solver, and couple the
fragments through a mean-field environment until the fragment density matrices are
self-consistent. The original QFlow papers use VQE on quantum hardware; this project
replaces VQE with TrimCI to stress-test the fragmentation on a system that is genuinely
hard classically.

### TrimCI

TrimCI is a selected configuration interaction solver. It builds the wavefunction
iteratively from a reference determinant using a heat-bath selection scheme, converging
to a compact wavefunction with far fewer determinants than full CI. For Fe4S4 with a
36-orbital active space, TrimCI converges at ~10,000 determinants where full CI would
require billions.

### Why Combine Them?

TrimCI still scales steeply with system size. The group's target systems have 50–80 orbital
active spaces where brute-force TrimCI is infeasible. Fragmentation into overlapping
~15-orbital windows keeps each sub-problem cheap regardless of total system size. This
project validates the fragmentation approach and quantifies the cost reduction on a system
where we can measure it exactly.

---

## What Is Being Compared?

This project compares three related but different ideas:

1. **Full TrimCI**
2. **Original QFlow**
3. **TrimCI-Flow**, the fragmented method in this repository

They can all start from the same molecular Hamiltonian, but they do not solve it in the
same way.

### The Shared Input: FCIDUMP

The file `data/fcidump_cycle_6` is the Hamiltonian input. It contains the one-electron
and two-electron integrals for the 36-orbital Fe4S4 active space:

```text
h1[p,q]
eri[p,q,r,s] = (pq|rs)
n_orb = 36
n_elec = 54
E_nuc = 0.0
```

Both full TrimCI and TrimCI-Flow read this same FCIDUMP. The difference is what they do
after reading it.

The FCIDUMP is not the final TrimCI wavefunction. It is the Hamiltonian that TrimCI
solves. The reference determinant archive `data/dets.npz` is output from a full TrimCI
calculation and is used here for reference electron counting and benchmarking.

### Full TrimCI: One Global Selected-CI Solve

Full TrimCI treats the Fe4S4 active space as one problem:

```text
same FCIDUMP
→ one 36-orbital / 54-electron Hamiltonian
→ selected-CI expansion over the full active space
→ determinants can excite across any orbitals globally
→ E_ref ≈ -327.1920 Ha
→ about 10,095 selected determinants
```

This is the most accurate calculation available in this project and is used as the
reference. It is still not full FCI in the literal sense; TrimCI is a selected-CI method.
But it is a full-active-space selected-CI solve, not a fragmented approximation.

For Fe4S4, the full FCI determinant space would be astronomically larger than 10,095
determinants. TrimCI makes the full active-space reference tractable by selecting only
the important determinants.

### Original QFlow: Many Small SES Fragments + Amplitude Iteration

Original QFlow is a fragment-based active-space method. In the QFlow papers, the active
space is broken into many small subsystem embedding subspaces, often called SES blocks.
For QFlow(4e,4o), each SES is a small active space built from combinations of occupied
and virtual orbitals.

The core QFlow idea is:

```text
enumerate many SES fragments
→ build an effective Hamiltonian for each SES
→ solve each SES, often with VQE or exact diagonalization
→ update a shared excitation-amplitude pool
→ rebuild effective Hamiltonians
→ iterate many times until the shared amplitude description converges
```

That is why even H-chain QFlow examples can involve many fragments and many optimization
iterations. The molecule is small, but the algorithm is amplitude-coupled and iterative.

For Fe4S4, a naive QFlow(4e,4o) enumeration would create far too many SES blocks because
there are 27 occupied spatial orbitals and only 9 virtual spatial orbitals per spin in
the reference determinant. The occupied-side combinatorics are the wall.

The current reference QFlow implementation also builds effective Hamiltonians using a
similarity transform in the full determinant space:

```text
H_eff = V^T exp(-sigma_ext) H exp(+sigma_ext) V
```

That full-space construction is not tractable for Fe4S4. This is why TrimCI-Flow does
not simply run the original QFlow code on this system.

### TrimCI-Flow: Coarse Fragmented TrimCI Embedding

TrimCI-Flow keeps the useful QFlow idea, but changes the implementation:

```text
same FCIDUMP
→ split the 36 orbitals into a few larger fragments
→ dress each fragment Hamiltonian with environment information
→ solve each fragment with TrimCI
→ assemble an approximate total energy
→ compare against the full TrimCI reference
```

The current Phase D method uses three non-overlapping 12-orbital fragments:

```text
F0: 12 orbitals
F1: 12 orbitals
F2: 12 orbitals
```

All fragments are included. In the h1-diagonal partition, F2 is closed-shell in the
reference determinant, so its local CI space has only one determinant. That means the
fragment is solved, but the solve is trivial.

The best Phase D result is:

```text
E_total = -326.568151 Ha
error vs full TrimCI reference = +0.623849 Ha
fragment_n_dets = [1008, 1008, 1]
total fragment determinants = 2017
```

This should not be described as solving the full Fe4S4 CI problem exactly. It is a
fragmented approximation to the full active-space problem. The full active space is
covered by fragments, but cross-fragment correlation is only approximated.

### Why TrimCI-Flow Runs So Much Faster

TrimCI-Flow is fast because it avoids one large global selected-CI solve and avoids the
many-SES iterative QFlow amplitude loop.

Instead of:

```text
one 36-orbital selected-CI expansion
```

or:

```text
many QFlow SES fragments × many amplitude iterations
```

Phase D does:

```text
three 12-orbital fragment TrimCI solves
one energy assembly step
no QFlow amplitude pool
no self-consistent amplitude loop
```

That is why a notebook can regenerate the best Phase D result in minutes. It is not
doing the same calculation as full TrimCI or original QFlow. It is doing a cheaper
fragmented approximation using TrimCI as the fragment solver.

### Cost-Accuracy Tradeoff

The important comparison is not "same exact solve, faster." The important comparison is:

> Same FCIDUMP input, cheaper fragmented approximation, measured against the full
> TrimCI reference.

For Fe4S4:

| Method | What is solved | Determinants | Energy (Ha) | Error vs −327.1920 |
|--------|----------------|-------------:|------------:|-------------------:|
| Full TrimCI reference | One full 36-orbital selected-CI problem | 10,095 | −327.1920 | 0.000 |
| Phase D loose D2 | Three 12-orbital embedded fragments | 147 | −326.511333 | +0.680667 |
| Phase D best D2 | Three 12-orbital embedded fragments | 2017 | −326.568151 | +0.623849 |

So the current result is:

```text
2017 / 10095 ≈ 20%
```

or about five times fewer selected determinants than the full TrimCI reference, with an
energy error of about `0.624 Ha`.

At the loose D2 setting:

```text
147 / 10095 ≈ 1.46%
```

or roughly 69 times fewer selected determinants, with an energy error of about `0.681 Ha`.

The research question is whether more QFlow-like cross-fragment coupling can reduce that
accuracy loss while keeping the determinant cost far below the full TrimCI reference.

### What TrimCI-Flow Does and Does Not Claim

TrimCI-Flow does claim:

- The full active space is covered by fragments.
- Every fragment is solved with TrimCI.
- The same FCIDUMP Hamiltonian is used as the full TrimCI reference.
- The fragmented method is much cheaper than the full selected-CI reference.
- The resulting energy can be benchmarked directly against full TrimCI.

TrimCI-Flow does not claim:

- It exactly solves the full 36-orbital active-space CI problem.
- It is already full QFlow amplitude coupling.
- Exact fragment solves automatically imply exact full-system energy.
- Summing fragment energies is valid without a defined embedding/correction formula.

The honest summary is:

> TrimCI-Flow is an end-to-end fragmented approximation solver. It uses the same
> FCIDUMP as full TrimCI, solves all orbital regions through fragments, and trades some
> accuracy for a large reduction in selected-determinant cost.

### Core Difficulties

- **Stochastic 1-RDM noise** — TrimCI has shot-to-shot variance in off-diagonal gamma
  elements that can destabilize mean-field loops. Convergence thresholds must account for
  this noise floor.
- **Fragment electron counts** — must use the correlated reference det from `dets.npz`,
  not the HF reference. HF assigns occupations rigidly and collapses fragments near the
  Fermi level to one-determinant solutions.
- **J − ½K dressing** — the mean-field correction uses spin-summed γ ∈ [0, 2], so the
  correct formula is J − ½K, not 2J − K (which would double the shift).
- **Cross-fragment correlation** — mean-field dressing only captures 1-RDM level
  environment. The dominant remaining error is two-body correlation between fragments.

---

## Phases

| Phase | Module | Method | Status |
|-------|--------|--------|--------|
| C | `uncoupled` | Independent fragment TrimCI, no coupling | **Complete, regression-locked** |
| D | `mfa` | Non-overlapping MFA: Fock-dressed h1 from diagonal γ | **Complete** |
| E | `mfa` | PT2 cross-fragment amplitude coupling (F0 ↔ F1) | **In progress** |

---

## Results

### Phase C — Uncoupled (baseline)

Overlapping sliding window partition (W=15, S=10 → 3 fragments), bare integrals, no coupling.

| Fragment | Orbitals | Determinants |
|----------|----------|-------------|
| F0 | 15 | 51 |
| F1 | 15 | 51 |
| F2 | 16 | 16 |
| **Total** | | **118** |

**118 total dets vs 10,095 brute-force = 1.2% of the reference cost.**

Regression-locked: `run_fragmented_trimci()` must always return `total_dets=118,
fragment_n_dets=[51, 51, 16]`.

### Phase D — MFA Non-Overlapping (current best)

Non-overlapping h1-diagonal partition (12 + 12 + 12 orbitals). Each fragment's h1 is
Fock-dressed by the diagonal 1-RDM from all other fragments (extracted from Phase B
overlapping convergence run).

Energy formula: `E_total = E_mf_global + Σ_I (E_TrimCI_I − E_mf_emb_I)`

| threshold | dets | E_total (Ha) | error vs −327.1920 |
|-----------|------|--------------|--------------------|
| 0.06 | **147** | −326.511 | +0.681 Ha |
| 0.02 | 1005 | −326.545 | +0.647 Ha |
| 0.01 | 2017 | −326.568 | +0.624 Ha |

- `E_mf_global = −323.080` Ha (partition-independent ✓, electron count 27α+27β ✓)
- Fragment 2 (orbs 24–35): fully closed-shell in reference → 1 determinant
- Energy converges slowly with threshold — cross-fragment correlation is the bottleneck

**Phase D v2 — off-diagonal gamma diagnostic:**  
Replacing diagonal γ with a full off-diagonal γ (from overlapping Phase B fragments)
changes `E_total` by only **+0.021 Ha** — well below the 0.05 Ha MFA ceiling threshold.
The 1-RDM dressing has hit its ceiling; the remaining +0.624 Ha gap is not recoverable by
improving the mean-field environment.

**Partition ablation:**  
Balanced partition (equal-size fragments) gives E = −326.411 Ha, +0.100 Ha worse than
h1-diagonal. Energy-block coherence (h1diag ordering) matters more than balance.

### Phase E — PT2 Cross-Fragment Coupling (in progress)

Tests whether Epstein-Nesbet PT2 double-excitation coupling between F0 and F1 recovers
energy beyond the Phase D baseline. Four spin channels (αα, ββ, αβ, βα), 3888 coupling
terms expected. F2 is fully closed-shell and inert.

**Pre-defined success criterion:**  
`|E_PT2_cross| > 0.1 Ha` AND `|ΔE_resolved| > 0.1 Ha` → promote to fuller BCH coupling.  
Both < 0.03 Ha → cross-fragment PT2 is not the missing piece at this partition.

---

## Key Numbers

| Quantity | Value |
|----------|-------|
| Brute-force energy | −327.1920 Ha |
| Brute-force dets | 10,095 |
| Phase C dets | 118 (1.2%) |
| Phase D best energy | −326.568 Ha (threshold=0.01, 2017 dets) |
| Phase D error | +0.624 Ha |
| MFA ceiling (Phase D v2) | +0.021 Ha off-diagonal effect — negligible |
| E_mf_global | −323.080 Ha |

---

## File Structure

```
TrimCI_Flow/
├── README.md
├── progress.md                    append-only experiment log (all runs)
├── TrimCI_Flow_Results.ipynb      executed results notebook (no TrimCI calls)
├── __init__.py
│
├── data/                          Fe4S4 input data (see data/README.md)
│   ├── fcidump_cycle_6            FCIDUMP integrals, 36 orbs, 54 electrons (9.2 MB)
│   ├── dets.npz                   10,095 reference dets, row 0 = correlated ref det
│   └── README.md                  dataset documentation
│
├── core/                          shared utilities (all phases)
│   ├── fragment.py                fragment_by_sliding_window, extract_fragment_integrals
│   ├── trimci_adapter.py          solve_fragment_trimci, FragmentResult
│   ├── results.py                 FragmentedRunResult dataclass
│   └── analysis.py                determinant_summary, convergence_summary
│
├── uncoupled/                     Phase C — regression-locked, do not modify
│   └── solver.py                  run_fragmented_trimci → 118 dets
│
├── mfa/                           Phase D + Phase E (active development)
│   ├── helpers.py                 compute_fragment_rdm1, dress_integrals_meanfield,
│   │                              assemble_global_rdm1_diag
│   ├── solver.py                  partition builders, h1 dressing, energy formula
│   ├── energy.py                  Fock, MFA global energy, embedded corrections
│   ├── extract_full_gamma.py      overlapping-fragment gamma extraction
│   └── runners/
│       ├── run_d1_overlapping.py      Phase D1: overlapping benchmark
│       └── run_d2_nonoverlapping.py   Phase D2: h1diag non-overlapping (canonical)
│
├── tests/
│   ├── test_mfa_energy.py
│   ├── test_mfa_solver.py
│   └── test_extract_full_gamma.py
│
└── Outputs/
    ├── uncoupled/                 Phase C output (regression reference)
    └── mfa/                       Phase D outputs + gamma .npy input files
        ├── outs_extract_full_gamma_*/     gamma_mixed_diag/full/hybrid.npy
        ├── outs_notebook_d1_overlapping_*/
        ├── outs_notebook_d2_h1diag_threshold006_*/   canonical baseline (147 dets)
        ├── outs_notebook_d2_h1diag_threshold001_*/   tight threshold (2017 dets)
        ├── outs_notebook_d2_balanced_*/               partition ablation
        ├── outs_d2_nonoverlapping_balanced_*/
        └── outs_d2_nonoverlapping_h1diag_*/           Phase D v2 hybrid gamma
```

---

## How to Run

```bash
source /home/unfunnypanda/Proj_Flow/qflowenv/bin/activate
cd /home/unfunnypanda/Proj_Flow

# Phase C regression check (must return 118 dets):
python -c "
from TrimCI_Flow.uncoupled import run_fragmented_trimci
from TrimCI_Flow.core import determinant_summary
r = run_fragmented_trimci(
    'Fe4S4_251230orbital_-327.1920_10kdets/Fe4S4_251230orbital_-327.1920_10kdets/fcidump_cycle_6')
determinant_summary(r)
assert r.total_dets == 118
print('Phase C regression: OK')
"

# Phase D canonical run (threshold=0.06, h1diag partition):
python -m TrimCI_Flow.mfa.runners.run_d2_nonoverlapping

# Tests (27 unit tests, no FCIDUMP required):
python -m pytest TrimCI_Flow/tests/ -v
```

---

## Reference Data

| Item | Path |
|------|------|
| FCIDUMP | `data/fcidump_cycle_6` (9.2 MB) |
| Reference dets | `data/dets.npz`, shape (10095, 2) uint64 |
| Diagonal gamma | `Outputs/mfa/outs_extract_full_gamma_*/gamma_mixed_diag.npy` |
| Full gamma | same dir, `gamma_mixed_full.npy` |

See `data/README.md` for full dataset documentation.

**Critical gotchas:**
- `E_nuc = 0.0` in this FCIDUMP — nuclear repulsion already absorbed. Do not add it.
- `dets.npz` row 0 is the correlated reference determinant — always use this for fragment
  electron counting, never the HF reference.
- `energy_from_rdm` binding: `h1` must be `.tolist()` (2D list), not `.ravel().tolist()`.
- Do NOT sum fragment energies — use the correlation-correction formula only.

---

## Archive

`archive/pre-cleanup` branch preserves all retired code (meanfield SCF, DMET 1-shot,
SC-DMET, prev_tests v2–v6, all historical outputs) for reference.
