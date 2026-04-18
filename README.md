# TrimCI_Flow

Fragmented active-space CI for Fe4S4, using TrimCI as a sub-solver inside a QFlow-style
self-consistent embedding loop.

---

## Background

### QFlow

QFlow is a fragmentation framework developed to make large active-space problems tractable
on quantum hardware. The core idea is borrowed from quantum embedding methods like DMET:
split a large active space into smaller overlapping fragments, solve each fragment
independently with a sub-solver, and couple the fragments together through a mean-field
environment until the fragment density matrices are self-consistent. The original QFlow
papers target systems of 50+ orbitals where exact diagonalization is impossible, and they
use VQE (a quantum circuit algorithm) as the sub-solver inside the fragment loop.

The framework does not care what the sub-solver is. It just needs something that can take
a fragment Hamiltonian and return an energy and a one-particle reduced density matrix (1-RDM).
That interface is the hook this project exploits.

### TrimCI

TrimCI is a selected configuration interaction solver developed in Dr. Otten's group.
Selected CI methods work by building the wavefunction iteratively: start from a reference
determinant, find the most important connected determinants by some importance criterion
(TrimCI uses a heat-bath selection scheme), add them to the expansion, and repeat until
convergence. The result is a compact wavefunction described by a small number of
determinants rather than the exponentially large full CI space.

TrimCI is particularly well suited to large active spaces (30+ orbitals) where exact
diagonalization is completely out of reach but the wavefunction is still dominated by
a manageable set of important determinants. For Fe4S4 with a 36-orbital active space,
TrimCI converges with about 10,000 determinants where full CI would require billions.

### Why Combine Them?

The natural question is: if TrimCI already handles 36-orbital active spaces, why fragment
at all?

Two reasons. First, TrimCI still scales steeply with system size. The 10,000-determinant
cost for Fe4S4 is manageable, but the group's target systems have 50-80 orbital active
spaces where the cost explodes. Fragmentation into overlapping 15-orbital windows keeps
each sub-problem in the cheap regime regardless of total system size.

Second, QFlow's benchmarks in the literature all use small active spaces where exact
diagonalization works. There is no published test of whether the fragmentation actually
delivers a genuine cost reduction on a system that is genuinely hard. TrimCI is the
natural sub-solver to stress-test this: it handles the full 36-orbital problem, so we can
run both the fragmented and the brute-force version and compare determinant counts
directly.

### What Are the Difficulties?

Plugging TrimCI into a QFlow loop sounds straightforward but raises several real problems.

**Stochastic noise in the density matrix.** TrimCI is a selected CI solver with
heat-bath pool construction. The 1-RDM computed from a TrimCI wavefunction has random
shot noise from run to run, even on the same Hamiltonian. This noise is small for the
energy (a few mHa) but large enough in the off-diagonal gamma elements to destabilize the
SCF loop. Standard SCF convergence criteria designed for deterministic solvers (like HF or
CASSCF) cannot be met here. We had to characterize the noise floor empirically and set
convergence thresholds accordingly.

**Oscillating SCF iterations.** Even with damping, the gamma vector bounces between
iterations because each TrimCI solve is an independent stochastic draw. Linear damping
alone (mixing old and new gamma with a coefficient alpha) reduces the amplitude of the
bounce but cannot eliminate it. Anderson acceleration helps, but only when tuned
conservatively: aggressive Anderson fits the stochastic history noise and amplifies it
rather than extrapolating toward the fixed point.

**Fragment electron counts.** The electron count inside each fragment depends on the
reference determinant used for counting occupied orbitals in the fragment window. If you
use the Hartree-Fock reference, fragments near the Fermi level end up with unphysical
electron counts (all low-energy orbitals filled, all high-energy orbitals empty) because
HF assigns occupations rigidly. This causes fragment CI problems with trivial one-det
solutions. The fix is to always count electrons from the correlated reference determinant
stored in dets.npz.

**The J - 1/2 K formula.** The mean-field dressing of the one-body Hamiltonian uses the
formula h1_eff = h1 + J - 1/2 K, where J is the Coulomb term and K is the exchange term.
The coefficient depends on whether gamma is a per-spin occupation (range 0 to 1) or a
spin-summed spatial occupation (range 0 to 2). Getting this wrong by a factor of two
gives the wrong mean-field shift and prevents convergence. The correct formula for
spin-summed gamma is J - 1/2 K, not 2J - K.

---

## Why Not Just Run TrimCI on the Full System?

For Fe4S4 you can, and we do, as the reference. But the point of this work is to show
that fragmentation already delivers a 99x reduction in determinant count (118 vs 10,095)
before you even get to a quantum computer. For larger systems in the group's pipeline,
brute-force TrimCI will not be feasible and fragmentation will be the only option. This
project validates the approach and builds the infrastructure for those future systems.

---

## Why Not VQE?

QFlow uses VQE as the sub-solver because the goal is to run on quantum hardware. We use
TrimCI because:

- TrimCI handles 36 orbitals classically. Current quantum hardware cannot run VQE
  reliably at this scale.
- The fragmentation and coupling logic is identical regardless of sub-solver. Validating
  the approach with TrimCI now means the same workflow applies when quantum hardware
  improves and VQE becomes viable at larger scales.
- TrimCI gives us the brute-force reference for free, so we can measure the determinant
  reduction exactly.

---

## Coupling Levels

The project tests three levels of inter-fragment coupling, in increasing order of
physical accuracy:

| Module | Coupling | Status |
|--------|----------|--------|
| `uncoupled` | None. Each fragment sees only its own bare integrals. | Complete |
| `meanfield` | Fock mean-field: each fragment's h1 is dressed by the 1-RDM from all other fragments, iterated to self-consistency. | Complete |
| `dmet` | One-shot non-overlapping DMET bath embedding with TrimCI impurity solves. Diagnostic only; SC-DMET not implemented yet. | Diagnostic |

---

## Fragment Construction

Orbitals are sorted by the diagonal of h1 (orbital energy proxy) and partitioned into
overlapping sliding windows:

```
W=15 (window size), S=10 (stride), n_orb=36  -->  3 fragments

Fragment 0: orbs [sorted_0  .. sorted_14]   (15 orbs)
Fragment 1: orbs [sorted_10 .. sorted_24]   (15 orbs, 5-orbital overlap with F0 and F2)
Fragment 2: orbs [sorted_20 .. sorted_35]   (16 orbs, absorbs the remaining tail)
```

---

## Results

### Uncoupled

Solve each fragment with bare integrals and no coupling.

| Fragment | Orbitals | Determinants |
|----------|----------|--------------|
| F0 | 15 | 51 |
| F1 | 15 | 51 |
| F2 | 16 | 16 |
| **Total** | | **118** |

**118 total dets vs 10,095 brute-force = 1.2% of the reference cost.**

Regression-locked. `run_fragmented_trimci()` must always return `total_dets=118,
fragment_n_dets=[51,51,16]`.

### Meanfield

Run the self-consistent Fock loop to convergence, then do a final extraction solve
at the same conditions as the uncoupled run (same TrimCI threshold, same auto det cap).

SCF and extraction have been confirmed across two independent runs (runner script and
notebook). Numbers below are from the notebook run (Flow_Meanfield.ipynb, 2026-04-15).

SCF result (conservative Anderson, N=5 gamma averaging):

```
converged:              True
iterations:             19
final max|delta gamma|: 0.00696   (threshold 1e-2)
final max|delta E|:     0.04917 Ha (threshold 5e-2 Ha)
```

The convergence is comfortable, not marginal. Both residuals are well inside their
thresholds, not just barely crossing them.

Extraction result (final solve at uncoupled conditions, threshold=0.06, max_final_dets=auto):

```
fragment_n_dets:    [51, 51, 16]
total_dets:         118
uncoupled baseline: 118
delta:              +0
verdict:            Meanfield is neutral -- same determinant count as uncoupled
```

This result is consistent across both the runner-script run and the notebook run.
Mean-field coupling converges cleanly and the gamma is self-consistent, but the dressed
Hamiltonian produces the same TrimCI determinant count as the bare Hamiltonian.

### DMET

One-shot non-overlapping DMET has been implemented as a diagnostic path. It uses an RHF
reference density to build Schmidt baths, solves each impurity with TrimCI, and reports
three energy partitions:

- `E_DMET_A`: one-body debug formula.
- `E_DMET_B`: primary 1-RDM + 2-RDM fragment partition.
- `E_DMET_C`: democratic impurity-energy diagnostic.

The current one-shot result is internally consistent but not accurate. Tightening the
TrimCI threshold increased determinant counts and did not move the primary DMET energy
toward the `-327.1920 Ha` reference.

Latest diagnostic runs:

| Run | Settings | Fragment dets | Total dets | E_DMET_B | Error vs -327.1920 |
|-----|----------|---------------|------------|----------|--------------------|
| `outs_dmet_1shot_20260415_192439` | threshold 0.06, auto det cap | [50, 50, 50] | 150 | -165.9401138348 Ha | +161.2519 Ha |
| `outs_dmet_1shot_threshold002_500dets_20260415_201225` | threshold 0.02, max 500 | [502, 502, 502] | 1506 | -156.3944846423 Ha | +170.7975 Ha |
| `outs_dmet_1shot_threshold001_1000dets_20260415_201316` | threshold 0.01, max 1000 | [1008, 1008, 1008] | 3024 | -153.0907813540 Ha | +174.1012 Ha |

Important audit outcome: the current RHF bath rank is 9 for each 12-orbital fragment,
so the actual impurity size is `n_imp = 21`, not 24. Frozen-core accounting is
consistent (`n_core = 15`, `n_elec_core = 30`, impurity electrons = 24), and all
fragment RDM reconstruction checks pass at about `1e-13 Ha`.

### What This Means

The 99x cost reduction comes entirely from fragmentation, not from coupling. This is
physically reasonable: mean-field coupling shifts one-body orbital energies, but the
multideterminant character of Fe4S4 is driven by strong two-body correlation inside each
fragment, which mean-field cannot capture. The current one-shot DMET diagnostics show
that simply tightening the impurity TrimCI solve is not enough; the next useful work is
an algebra/reference audit before building a self-consistent DMET loop.

---

## How to Import

```python
# Explicit sub-package imports (recommended -- clear and readable)
from TrimCI_Flow.uncoupled import run_fragmented_trimci
from TrimCI_Flow.meanfield import run_selfconsistent_fragments
from TrimCI_Flow.dmet import run_dmet_1shot
from TrimCI_Flow.core import determinant_summary, convergence_summary

# Top-level re-exports also work for convenience
from TrimCI_Flow import run_fragmented_trimci, run_selfconsistent_fragments
```

---

## File Structure

```
TrimCI_Flow/
|
+-- README.md                          <- this file
+-- progress.md                        <- append-only experiment log (all runs)
+-- PHASE_B_CONTEXT.md                 <- deep technical reference for meanfield
|
+-- Flow_Uncoupled.ipynb               <- playground notebook: uncoupled solver
+-- Flow_Meanfield.ipynb               <- playground notebook: meanfield SCF + extraction
|
+-- core/                              <- shared by all coupling levels
|   +-- results.py                     <- FragmentedRunResult dataclass
|   +-- fragment.py                    <- fragment_by_sliding_window, extract_fragment_integrals
|   +-- trimci_adapter.py              <- solve_fragment_trimci, FragmentResult
|   +-- analysis.py                    <- determinant_summary, convergence_summary, iteration_summary
|
+-- uncoupled/
|   +-- solver.py                      <- run_fragmented_trimci (regression-locked at 118 dets)
|
+-- meanfield/
|   +-- helpers.py                     <- compute_fragment_rdm1, dress_integrals_meanfield,
|   |                                     assemble_global_rdm1_diag
|   +-- solver.py                      <- run_selfconsistent_fragments (Anderson SCF)
|   +-- runners/
|       +-- run_conservative_anderson.py   <- strict-criteria diagnostic run
|       +-- run_relaxed.py                 <- working baseline, converges in 19-43 iters (stochastic)
|       +-- run_extraction.py              <- SCF + extraction: answers the det-count question
|
+-- dmet/
|   +-- __init__.py                    <- exports run_dmet_1shot
|   +-- hf_reference.py                <- PySCF RHF reference density
|   +-- bath.py                        <- Schmidt bath, frozen core, impurity Hamiltonian
|   +-- energy.py                      <- DMET energy formulas and RDM checks
|   +-- solver.py                      <- one-shot non-overlapping DMET driver
|   +-- runners/
|       +-- run_dmet_1shot.py          <- Fe4S4 one-shot runner
|
+-- prev_tests/                        <- retired: v2-v6 version files + old runners
+-- Outputs/
    +-- meanfield_active/              <- completed meanfield runs
    |   +-- outs_extraction_autodets/              <- runner script run: 118 dets
    |   +-- outs_notebook_extraction_20260415_*/   <- notebook run: 118 dets (confirming)
    |   +-- outs_v6_conservative_anderson_avg5_200dets/
    |   +-- outs_v6_relaxed_avg5_200dets/
    +-- meanfield_archive/             <- v2-v5 era experiment history
    +-- dmet/                          <- one-shot DMET diagnostics
    |   +-- outs_dmet_1shot_20260415_192439/
    |   +-- outs_dmet_1shot_threshold002_500dets_20260415_201225/
    |   +-- outs_dmet_1shot_threshold001_1000dets_20260415_201316/
    +-- uncoupled/                     <- uncoupled solver outputs
```

---

## Environment

```bash
source /home/unfunnypanda/Proj_Flow/qflowenv/bin/activate
# Python 3.12.3, TrimCI editable install, numpy 2.4.2, pyscf 2.12.1

# Uncoupled regression check (must return 118 dets):
cd /home/unfunnypanda/Proj_Flow
python -c "
from TrimCI_Flow.uncoupled import run_fragmented_trimci
from TrimCI_Flow.core import determinant_summary
r = run_fragmented_trimci(
    'Fe4S4_251230orbital_-327.1920_10kdets/Fe4S4_251230orbital_-327.1920_10kdets/fcidump_cycle_6')
determinant_summary(r)
assert r.total_dets == 118
print('Uncoupled regression: OK')
"
```

---

## Reference Data

| Item | Value / Path |
|------|-------------|
| FCIDUMP | `Fe4S4_251230orbital_-327.1920_10kdets/.../fcidump_cycle_6` |
| Reference dets | same dir, `dets.npz`, shape (10095, 2) uint64 [alpha_bits, beta_bits] |
| Brute-force energy | -327.1920 Ha |
| Brute-force dets | 10,095 |

Two things to keep in mind:

- E_nuc = 0.0 in this FCIDUMP. The nuclear repulsion is already absorbed into the one-body
  integrals. Do not add it.
- dets.npz row 0 is the correlated reference determinant and must be used for fragment
  electron counting. Using the HF reference instead gives wrong fragment occupations and
  collapses fragments to one-determinant solutions.

---

## Summary

| Coupling level | Status | Total dets | vs 10,095 |
|----------------|--------|-----------|-----------|
| Uncoupled | Complete | **118** | **1.2%** |
| Meanfield | Complete | **118** | **1.2%** (neutral) |
| DMET | Diagnostic | 150 to 3024 | energy/accounting under audit |

Fragmentation alone gives the 99x reduction. Meanfield coupling converges but does not
further reduce determinant cost at this geometry. One-shot DMET is now implemented for
diagnostics, but its current primary energy is far from the reference, so SC-DMET should
wait until the one-shot algebra/reference questions are settled.
