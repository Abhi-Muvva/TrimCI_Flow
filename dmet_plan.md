# DMET v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement 1-shot non-overlapping DMET with TrimCI as impurity solver, producing a physically meaningful total energy E_DMET from a proper 1-RDM + 2-RDM fragment energy partition.

**Architecture:** PySCF RHF generates a mean-field reference γ_MF (36×36). For each of three non-overlapping 12-orbital fragments (partitioned by h1-diagonal ordering), Schmidt decomposition of γ_MF builds a 12-orbital bath via SVD of the frag–env block; the SVD orthogonal complement is the frozen core. TrimCI solves the resulting 24-orbital impurity Hamiltonian; `compute_2rdm` extracts Γ2; three energy formulas (A: 1-body only, B: primary 2-RDM partition, C: democratic) partition the total energy as E_DMET = E_nuc + Σ_I E_I^frag.

**Tech Stack:** Python 3.12.3, NumPy 2.4.2, PySCF 2.12.1, TrimCI (editable install). Activate env: `source /home/unfunnypanda/Proj_Flow/qflowenv/bin/activate`. Run all commands from `/home/unfunnypanda/Proj_Flow/`.

---

## Spec reference

`docs/superpowers/specs/2026-04-15-dmet-design.md` — read it before starting.

## Critical constraints (read before touching any file)

- `uncoupled/`, `meanfield/`, `core/`, `prev_tests/` are **READ-ONLY**. Zero modifications.
- `progress.md` is **append-only**. Every run appends a `# DMET — Run N` section.
- Phase C regression: `run_fragmented_trimci` must still give `total_dets == 118`, `fragment_n_dets == [51, 51, 16]`.
- `E_nuc = 0.0` (already absorbed into FCIDUMP). Do not add it.
- `compute_1rdm`, `compute_2rdm`, `energy_from_rdm` from `trimci.trimci_core` — do not reimplement.
- Approach B energy uses `h1_solver` (includes v_fock_core), NOT `h1_phys_proj`. See spec §4.7.
- `energy_from_rdm` binding: `h1` is `std::vector<std::vector<double>>` → pass `.tolist()` (2D). `eri`, `gamma`, `gamma2` are flat → pass `.ravel().tolist()`.

---

## File map

```
# New files (all relative to /home/unfunnypanda/Proj_Flow/)
TrimCI_Flow/tests/__init__.py                      ← empty, marks test package
TrimCI_Flow/tests/test_dmet_hf_reference.py        ← Task 1 unit tests
TrimCI_Flow/tests/test_dmet_bath.py                ← Tasks 2–3 unit tests
TrimCI_Flow/tests/test_dmet_energy.py              ← Task 4 unit tests
TrimCI_Flow/dmet/hf_reference.py                   ← Task 1: run_hf()
TrimCI_Flow/dmet/bath.py                           ← Tasks 2–3: schmidt_decomp(),
                                                      build_impurity_hamiltonian(),
                                                      impurity_electron_count()
TrimCI_Flow/dmet/energy.py                         ← Task 4: check_2rdm_convention(),
                                                      dmet_energy_a/b/c()
TrimCI_Flow/dmet/solver.py                         ← Task 5: run_dmet_1shot()
TrimCI_Flow/dmet/runners/__init__.py               ← Task 6: empty
TrimCI_Flow/dmet/runners/run_dmet_1shot.py         ← Task 6: top-level runner

# Modified files
TrimCI_Flow/dmet/__init__.py                       ← Task 6: export run_dmet_1shot
TrimCI_Flow/progress.md                            ← Task 8: append DMET run entry
```

---

## Task 0: Test infrastructure

**Files:**
- Create: `TrimCI_Flow/tests/__init__.py`

- [ ] **Step 1: Create the test package init**

```bash
touch /home/unfunnypanda/Proj_Flow/TrimCI_Flow/tests/__init__.py
```

- [ ] **Step 2: Verify imports work from project root**

```bash
cd /home/unfunnypanda/Proj_Flow
source qflowenv/bin/activate
python -c "import TrimCI_Flow; import trimci; import pyscf; print('OK')"
```

Expected output: `OK`

- [ ] **Step 3: Verify pytest is available**

```bash
python -m pytest --version
```

Expected: `pytest X.Y.Z`

---

## Task 1: HF reference module

**Files:**
- Create: `TrimCI_Flow/dmet/hf_reference.py`
- Create: `TrimCI_Flow/tests/test_dmet_hf_reference.py`

- [ ] **Step 1: Write failing tests**

```python
# TrimCI_Flow/tests/test_dmet_hf_reference.py
import numpy as np
import pytest


def _tiny_system():
    """2-orbital, 2-electron toy system with known closed-shell HF solution."""
    n_orb, n_elec = 2, 2
    h1 = np.array([[-1.0, -0.1],
                   [-0.1, -0.5]])
    eri = np.zeros((2, 2, 2, 2))
    eri[0, 0, 0, 0] = 0.6
    eri[1, 1, 1, 1] = 0.4
    eri[0, 0, 1, 1] = eri[1, 1, 0, 0] = 0.3
    eri[0, 1, 0, 1] = eri[1, 0, 1, 0] = 0.1
    eri[0, 1, 1, 0] = eri[1, 0, 0, 1] = 0.1
    return h1, eri, n_elec, n_orb


def test_gamma_mf_shape_and_trace():
    from TrimCI_Flow.dmet.hf_reference import run_hf
    h1, eri, n_elec, n_orb = _tiny_system()
    gamma_mf, e_hf = run_hf(h1, eri, n_elec, n_orb)
    assert gamma_mf.shape == (n_orb, n_orb)
    assert np.isclose(np.trace(gamma_mf), n_elec, atol=1e-8), \
        f"Tr(gamma_mf) = {np.trace(gamma_mf):.6f}, expected {n_elec}"


def test_gamma_mf_is_symmetric():
    from TrimCI_Flow.dmet.hf_reference import run_hf
    h1, eri, n_elec, n_orb = _tiny_system()
    gamma_mf, _ = run_hf(h1, eri, n_elec, n_orb)
    np.testing.assert_allclose(gamma_mf, gamma_mf.T, atol=1e-10)


def test_e_hf_is_negative_float():
    from TrimCI_Flow.dmet.hf_reference import run_hf
    h1, eri, n_elec, n_orb = _tiny_system()
    _, e_hf = run_hf(h1, eri, n_elec, n_orb)
    assert isinstance(e_hf, float)
    assert e_hf < 0, f"Expected negative RHF energy, got {e_hf}"


def test_no_convergence_warning_on_easy_system():
    """A simple diagonal h1 system should converge without warning."""
    import io, sys
    from TrimCI_Flow.dmet.hf_reference import run_hf
    n_orb, n_elec = 2, 2
    h1 = np.diag([-1.0, 0.5])
    eri = np.zeros((2, 2, 2, 2))
    eri[0, 0, 0, 0] = 0.4; eri[0, 0, 1, 1] = eri[1, 1, 0, 0] = 0.2; eri[1, 1, 1, 1] = 0.3
    buf = io.StringIO()
    sys.stdout = buf
    run_hf(h1, eri, n_elec, n_orb)
    sys.stdout = sys.__stdout__
    assert "WARNING" not in buf.getvalue(), \
        f"Unexpected convergence warning: {buf.getvalue()}"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /home/unfunnypanda/Proj_Flow
python -m pytest TrimCI_Flow/tests/test_dmet_hf_reference.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'TrimCI_Flow.dmet.hf_reference'`

- [ ] **Step 3: Implement hf_reference.py**

```python
# TrimCI_Flow/dmet/hf_reference.py
"""
hf_reference.py
===============
Run PySCF RHF on the active-space Hamiltonian from a FCIDUMP to obtain
the mean-field 1-RDM (gamma_mf) used for DMET bath construction.
"""
from __future__ import annotations

import numpy as np


def run_hf(
    h1: np.ndarray,
    eri: np.ndarray,
    n_elec: int,
    n_orb: int,
) -> tuple[np.ndarray, float]:
    """
    Run PySCF RHF on the active-space Hamiltonian.

    Parameters
    ----------
    h1     : (n_orb, n_orb) one-body integrals (chemist notation)
    eri    : (n_orb, n_orb, n_orb, n_orb) two-electron integrals (chemist notation)
    n_elec : total number of electrons (n_alpha + n_beta; must be even)
    n_orb  : number of spatial orbitals

    Returns
    -------
    gamma_mf : (n_orb, n_orb) spin-summed 1-RDM from RHF, values in [0, 2]
    e_hf     : float, RHF total energy (E_nuc = 0 in FCIDUMP convention)

    Warns (print, not raise) if RHF does not converge. The density matrix is
    still returned and used for bath construction.
    """
    from pyscf import gto, scf, ao2mo

    mol = gto.M()
    mol.nelectron = n_elec
    mol.spin      = 0           # n_alpha == n_beta (closed-shell)
    mol.verbose   = 0

    mf = scf.RHF(mol)
    mf.get_hcore = lambda *_: h1
    mf.get_ovlp  = lambda *_: np.eye(n_orb)
    mf._eri      = ao2mo.restore(8, eri, n_orb)
    mf.kernel()

    if not mf.converged:
        print("[DMET WARNING] RHF did not converge — gamma_mf may be unreliable")

    gamma_mf = mf.make_rdm1()   # shape (n_orb, n_orb), spin-summed
    return gamma_mf, float(mf.e_tot)
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
python -m pytest TrimCI_Flow/tests/test_dmet_hf_reference.py -v
```

Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add TrimCI_Flow/dmet/hf_reference.py TrimCI_Flow/tests/__init__.py TrimCI_Flow/tests/test_dmet_hf_reference.py
git commit -m "feat(dmet): add hf_reference.py — PySCF RHF → gamma_mf"
```

---

## Task 2: Bath construction — schmidt_decomp

**Files:**
- Create: `TrimCI_Flow/dmet/bath.py` (partial — schmidt_decomp only)
- Create: `TrimCI_Flow/tests/test_dmet_bath.py` (partial)

- [ ] **Step 1: Write failing tests for schmidt_decomp**

```python
# TrimCI_Flow/tests/test_dmet_bath.py
import numpy as np
import pytest


# ── Fixtures ────────────────────────────────────────────────────────────────

def _gamma_4orb():
    """4-orbital gamma_mf with frag=[0,1], env=[2,3].
    gamma_AB (2×2) has full rank → n_bath=2, n_core=0."""
    return np.array([
        [1.5, 0.3, 0.4, 0.1],
        [0.3, 0.8, 0.2, 0.1],
        [0.4, 0.2, 1.2, 0.3],
        [0.1, 0.1, 0.3, 0.9],
    ]), [0, 1], 4


def _gamma_6orb_with_core():
    """6-orbital gamma_mf with frag=[0,1], env=[2,3,4,5].
    gamma_AB (2×4) has rank 2 → n_bath=2, n_core=2."""
    g = np.eye(6)
    g[0, 2] = g[2, 0] = 0.4
    g[1, 3] = g[3, 1] = 0.3
    g[2, 3] = g[3, 2] = 0.1
    g[4, 5] = g[5, 4] = 0.1
    return g, [0, 1], 6


# ── Tests ───────────────────────────────────────────────────────────────────

def test_env_orbs_returned_correctly():
    from TrimCI_Flow.dmet.bath import schmidt_decomp
    gamma, frag_orbs, n_orb = _gamma_4orb()
    _, _, _, _, env_orbs = schmidt_decomp(gamma, frag_orbs, n_orb)
    assert env_orbs == [2, 3]


def test_p_imp_columns_orthonormal_no_core():
    from TrimCI_Flow.dmet.bath import schmidt_decomp
    gamma, frag_orbs, n_orb = _gamma_4orb()
    P_imp, _, _, n_bath, _ = schmidt_decomp(gamma, frag_orbs, n_orb)
    n_imp = len(frag_orbs) + n_bath
    assert P_imp.shape == (n_orb, n_imp)
    np.testing.assert_allclose(P_imp.T @ P_imp, np.eye(n_imp), atol=1e-10,
                                err_msg="P_imp columns not orthonormal")


def test_fragment_identity_block_in_p_imp():
    """Columns 0:nfrag of P_imp must be identity at frag_orbs positions."""
    from TrimCI_Flow.dmet.bath import schmidt_decomp
    gamma, frag_orbs, n_orb = _gamma_4orb()
    P_imp, _, _, _, _ = schmidt_decomp(gamma, frag_orbs, n_orb)
    nfrag = len(frag_orbs)
    frag_block = P_imp[np.ix_(frag_orbs, list(range(nfrag)))]
    np.testing.assert_allclose(frag_block, np.eye(nfrag), atol=1e-10)


def test_n_bath_full_rank_env():
    """4-orbital: frag-env block is 2×2 full rank → n_bath == 2."""
    from TrimCI_Flow.dmet.bath import schmidt_decomp
    gamma, frag_orbs, n_orb = _gamma_4orb()
    _, _, _, n_bath, _ = schmidt_decomp(gamma, frag_orbs, n_orb)
    assert n_bath == 2


def test_gamma_core_full_zeros_when_no_core():
    """With n_core=0, gamma_core_full must be the zero matrix."""
    from TrimCI_Flow.dmet.bath import schmidt_decomp
    gamma, frag_orbs, n_orb = _gamma_4orb()
    _, gamma_core_full, n_elec_core, _, _ = schmidt_decomp(gamma, frag_orbs, n_orb)
    np.testing.assert_allclose(gamma_core_full, np.zeros((n_orb, n_orb)), atol=1e-10)
    assert np.isclose(n_elec_core, 0.0, atol=1e-10)


def test_gamma_core_full_symmetric_with_core():
    """6-orbital with core: gamma_core_full is symmetric and non-zero."""
    from TrimCI_Flow.dmet.bath import schmidt_decomp
    gamma, frag_orbs, n_orb = _gamma_6orb_with_core()
    _, gamma_core_full, n_elec_core, n_bath, _ = schmidt_decomp(gamma, frag_orbs, n_orb)
    assert n_bath == 2
    np.testing.assert_allclose(gamma_core_full, gamma_core_full.T, atol=1e-10,
                                err_msg="gamma_core_full not symmetric")
    assert np.linalg.norm(gamma_core_full) > 1e-10, "expected non-zero core density"
    assert n_elec_core > 1e-10


def test_gamma_core_zero_in_fragment_block():
    """gamma_core_full is zero in the fragment orbital block."""
    from TrimCI_Flow.dmet.bath import schmidt_decomp
    gamma, frag_orbs, n_orb = _gamma_6orb_with_core()
    _, gamma_core_full, _, _, _ = schmidt_decomp(gamma, frag_orbs, n_orb)
    frag_block = gamma_core_full[np.ix_(frag_orbs, frag_orbs)]
    np.testing.assert_allclose(frag_block, np.zeros((len(frag_orbs), len(frag_orbs))),
                                atol=1e-10)


def test_electron_conservation_with_core():
    """n_elec_imp + n_elec_core == n_elec_total for each fragment."""
    from TrimCI_Flow.dmet.bath import schmidt_decomp
    gamma, frag_orbs, n_orb = _gamma_6orb_with_core()
    _, _, n_elec_core, _, _ = schmidt_decomp(gamma, frag_orbs, n_orb)
    n_elec_total = float(np.trace(gamma))
    n_elec_imp = n_elec_total - n_elec_core
    assert np.isclose(n_elec_imp + n_elec_core, n_elec_total, atol=1e-8)


def test_p_imp_orthonormal_with_core():
    from TrimCI_Flow.dmet.bath import schmidt_decomp
    gamma, frag_orbs, n_orb = _gamma_6orb_with_core()
    P_imp, _, _, n_bath, _ = schmidt_decomp(gamma, frag_orbs, n_orb)
    n_imp = len(frag_orbs) + n_bath
    assert P_imp.shape == (n_orb, n_imp)
    np.testing.assert_allclose(P_imp.T @ P_imp, np.eye(n_imp), atol=1e-10)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest TrimCI_Flow/tests/test_dmet_bath.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'TrimCI_Flow.dmet.bath'`

- [ ] **Step 3: Implement schmidt_decomp in bath.py**

```python
# TrimCI_Flow/dmet/bath.py
"""
bath.py
=======
Schmidt decomposition, impurity Hamiltonian construction, and electron counting
for DMET 1-shot non-overlapping embedding.
"""
from __future__ import annotations

import numpy as np


def schmidt_decomp(
    gamma_mf: np.ndarray,
    frag_orbs: list[int],
    n_orb: int,
) -> tuple[np.ndarray, np.ndarray, float, int, list[int]]:
    """
    Schmidt decompose gamma_mf for the given fragment.

    Parameters
    ----------
    gamma_mf  : (n_orb, n_orb) spin-summed 1-RDM from RHF
    frag_orbs : orbital indices belonging to this fragment (in full-system space)
    n_orb     : total number of orbitals (36 for Fe4S4)

    Returns
    -------
    P_imp          : (n_orb, n_imp) projection matrix.
                     Columns 0:n_frag = fragment identity block.
                     Columns n_frag:n_frag+n_bath = bath vectors in env space.
    gamma_core_full: (n_orb, n_orb) frozen core density matrix.
                     Non-zero only in the env_orbs block. Built from the SVD
                     orthogonal complement (NOT env_orbs[n_bath:]).
    n_elec_core    : float, Tr(gamma_core) — electrons in the frozen core.
    n_bath         : int, number of significant bath orbitals.
    env_orbs       : list[int], environment orbital indices (full-system space).
    """
    frag_set = set(frag_orbs)
    env_orbs = [r for r in range(n_orb) if r not in frag_set]
    n_frag   = len(frag_orbs)
    n_env    = len(env_orbs)

    # frag–env block of gamma_mf: (n_frag, n_env)
    gamma_AB = gamma_mf[np.ix_(frag_orbs, env_orbs)]

    # Full SVD — Vt is (n_env, n_env)
    # Rows 0:n_bath of Vt → bath subspace
    # Rows n_bath:   of Vt → orthogonal complement (core)
    _, s, Vt = np.linalg.svd(gamma_AB, full_matrices=True)
    n_bath = int(np.sum(s > 1e-10))
    n_core = n_env - n_bath

    bath_vecs_env = Vt[:n_bath, :].T   # (n_env, n_bath) in env-orbital space
    core_vecs_env = Vt[n_bath:,  :].T  # (n_env, n_core) orthogonal complement

    # Projection matrix P_imp: (n_orb, n_frag + n_bath)
    n_imp = n_frag + n_bath
    P_imp = np.zeros((n_orb, n_imp))
    for i, f in enumerate(frag_orbs):
        P_imp[f, i] = 1.0                              # fragment identity
    for i, e in enumerate(env_orbs):
        P_imp[e, n_frag:] = bath_vecs_env[i, :]        # bath vectors

    # Core density matrix
    gamma_env        = gamma_mf[np.ix_(env_orbs, env_orbs)]           # (n_env, n_env)
    gamma_core_local = core_vecs_env.T @ gamma_env @ core_vecs_env    # (n_core, n_core)

    # Reconstruct in full orbital space — non-zero only in env_orbs block
    gamma_core_env_blk = core_vecs_env @ gamma_core_local @ core_vecs_env.T  # (n_env, n_env)
    gamma_core_full = np.zeros((n_orb, n_orb))
    gamma_core_full[np.ix_(env_orbs, env_orbs)] = gamma_core_env_blk

    n_elec_core = float(np.trace(gamma_core_local))

    return P_imp, gamma_core_full, n_elec_core, n_bath, env_orbs
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest TrimCI_Flow/tests/test_dmet_bath.py -v
```

Expected: all `test_*schmidt*` tests pass (8 tests).

- [ ] **Step 5: Commit**

```bash
git add TrimCI_Flow/dmet/bath.py TrimCI_Flow/tests/test_dmet_bath.py
git commit -m "feat(dmet): schmidt_decomp with orthogonal-complement core density"
```

---

## Task 3: Bath construction — impurity Hamiltonian + electron count

**Files:**
- Modify: `TrimCI_Flow/dmet/bath.py` (append two functions)
- Modify: `TrimCI_Flow/tests/test_dmet_bath.py` (append tests)

- [ ] **Step 1: Append tests for build_impurity_hamiltonian and impurity_electron_count**

Append to `TrimCI_Flow/tests/test_dmet_bath.py`:

```python
# ── build_impurity_hamiltonian tests ─────────────────────────────────────────

def _build_tiny_imp():
    """Minimal 4-orbital system for impurity Hamiltonian tests."""
    from TrimCI_Flow.dmet.bath import schmidt_decomp
    n_orb = 4
    frag_orbs = [0, 1]
    gamma_mf = np.array([
        [1.5, 0.3, 0.4, 0.1],
        [0.3, 0.8, 0.2, 0.1],
        [0.4, 0.2, 1.2, 0.3],
        [0.1, 0.1, 0.3, 0.9],
    ])
    h1 = np.arange(16, dtype=float).reshape(4, 4)
    h1 = (h1 + h1.T) / 2  # symmetrize
    eri = np.zeros((4, 4, 4, 4))
    # Minimal Coulomb integrals
    for p in range(4):
        eri[p, p, p, p] = 0.5
        for q in range(4):
            if p != q:
                eri[p, p, q, q] = eri[q, q, p, p] = 0.2
                eri[p, q, p, q] = eri[q, p, q, p] = 0.1
    P_imp, gamma_core_full, n_elec_core, n_bath, env_orbs = schmidt_decomp(
        gamma_mf, frag_orbs, n_orb)
    return h1, eri, gamma_mf, gamma_core_full, env_orbs, P_imp, frag_orbs, n_orb


def test_h1_phys_proj_matches_manual():
    """h1_phys_proj must equal P_imp.T @ h1 @ P_imp."""
    from TrimCI_Flow.dmet.bath import build_impurity_hamiltonian
    h1, eri, _, gamma_core_full, env_orbs, P_imp, _, _ = _build_tiny_imp()
    h1_phys, h1_sol, _ = build_impurity_hamiltonian(h1, eri, gamma_core_full, env_orbs, P_imp)
    expected = P_imp.T @ h1 @ P_imp
    np.testing.assert_allclose(h1_phys, expected, atol=1e-10,
                                err_msg="h1_phys_proj != P.T @ h1 @ P")


def test_h1_solver_includes_fock_correction():
    """h1_solver = h1_phys_proj + v_fock; for zero core, they must be equal."""
    from TrimCI_Flow.dmet.bath import build_impurity_hamiltonian
    # 4-orbital: n_core=0 → v_fock=0 → h1_solver == h1_phys_proj
    h1, eri, _, gamma_core_full, env_orbs, P_imp, _, _ = _build_tiny_imp()
    h1_phys, h1_sol, _ = build_impurity_hamiltonian(h1, eri, gamma_core_full, env_orbs, P_imp)
    # gamma_core_full is zero (n_core=0 for 4-orbital system), so v_fock=0
    np.testing.assert_allclose(h1_sol, h1_phys, atol=1e-10,
                                err_msg="v_fock non-zero with zero core density")


def test_h1_solver_fock_nonzero_with_core():
    """With a nonzero core density, h1_solver must differ from h1_phys_proj."""
    from TrimCI_Flow.dmet.bath import schmidt_decomp, build_impurity_hamiltonian
    n_orb = 6
    frag_orbs = [0, 1]
    gamma_mf = np.eye(6)
    gamma_mf[0, 2] = gamma_mf[2, 0] = 0.4
    gamma_mf[1, 3] = gamma_mf[3, 1] = 0.3
    gamma_mf[2, 3] = gamma_mf[3, 2] = 0.1
    gamma_mf[4, 5] = gamma_mf[5, 4] = 0.1
    h1 = np.eye(6) * -1.0
    eri = np.zeros((6, 6, 6, 6))
    for p in range(6):
        eri[p, p, p, p] = 0.5
        for q in range(p+1, 6):
            eri[p,p,q,q] = eri[q,q,p,p] = 0.2
            eri[p,q,p,q] = eri[q,p,q,p] = 0.1
    P_imp, gamma_core_full, _, _, env_orbs = schmidt_decomp(gamma_mf, frag_orbs, n_orb)
    h1_phys, h1_sol, _ = build_impurity_hamiltonian(h1, eri, gamma_core_full, env_orbs, P_imp)
    # Core is non-zero → h1_solver must differ from h1_phys_proj
    diff = np.max(np.abs(h1_sol - h1_phys))
    assert diff > 1e-10, f"Expected nonzero Fock correction; max|diff|={diff:.2e}"


def test_eri_proj_shape_and_symmetry():
    """eri_proj must be (n_imp, n_imp, n_imp, n_imp) and preserve chemist symmetry."""
    from TrimCI_Flow.dmet.bath import build_impurity_hamiltonian
    h1, eri, _, gamma_core_full, env_orbs, P_imp, _, _ = _build_tiny_imp()
    _, _, eri_proj = build_impurity_hamiltonian(h1, eri, gamma_core_full, env_orbs, P_imp)
    n_imp = P_imp.shape[1]
    assert eri_proj.shape == (n_imp, n_imp, n_imp, n_imp)
    # Chemist symmetry: (pq|rs) = (qp|rs) = (pq|sr) = (rs|pq)
    np.testing.assert_allclose(eri_proj, eri_proj.transpose(1, 0, 2, 3), atol=1e-10)
    np.testing.assert_allclose(eri_proj, eri_proj.transpose(0, 1, 3, 2), atol=1e-10)
    np.testing.assert_allclose(eri_proj, eri_proj.transpose(2, 3, 0, 1), atol=1e-10)


# ── impurity_electron_count tests ────────────────────────────────────────────

def test_electron_count_even_total():
    from TrimCI_Flow.dmet.bath import impurity_electron_count
    n_alpha, n_beta = impurity_electron_count(n_elec_total=54, n_elec_core=12.0)
    assert n_alpha == n_beta == 21
    assert n_alpha + n_beta == 42


def test_electron_count_raises_on_odd():
    from TrimCI_Flow.dmet.bath import impurity_electron_count
    with pytest.raises(AssertionError):
        impurity_electron_count(n_elec_total=54, n_elec_core=11.0)  # 43 → odd
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest TrimCI_Flow/tests/test_dmet_bath.py::test_h1_phys_proj_matches_manual -v 2>&1 | tail -5
```

Expected: `AttributeError` or `ImportError` (functions not yet defined).

- [ ] **Step 3: Append build_impurity_hamiltonian and impurity_electron_count to bath.py**

Append to `TrimCI_Flow/dmet/bath.py` (after `schmidt_decomp`):

```python
def build_impurity_hamiltonian(
    h1: np.ndarray,
    eri: np.ndarray,
    gamma_core_full: np.ndarray,
    env_orbs: list[int],
    P_imp: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project h1 and eri to the impurity space and add the Fock embedding
    correction from the frozen core density.

    Parameters
    ----------
    h1              : (n_orb, n_orb) physical one-body integrals
    eri             : (n_orb, n_orb, n_orb, n_orb) physical ERIs (chemist notation)
    gamma_core_full : (n_orb, n_orb) frozen core density (from schmidt_decomp).
                      Non-zero only in the env_orbs block.
    env_orbs        : environment orbital indices (returned by schmidt_decomp)
    P_imp           : (n_orb, n_imp) projection matrix (from schmidt_decomp)

    Returns
    -------
    h1_phys_proj : (n_imp, n_imp) physical h1 in impurity space — NO Fock correction.
                   Used in energy formula A and as the baseline for A vs B comparison.
    h1_solver    : (n_imp, n_imp) h1_phys_proj + v_fock_core_imp.
                   Used as input to TrimCI AND in energy formulas B and C.
    eri_proj     : (n_imp, n_imp, n_imp, n_imp) physical ERIs in impurity space.
                   Used in both TrimCI and energy formulas B and C.
    """
    n_orb = h1.shape[0]

    # Physical h1 projected to impurity
    h1_phys_proj = P_imp.T @ h1 @ P_imp   # (n_imp, n_imp)

    # ERI projection: four sequential contractions (cheapest order)
    T = np.einsum('pqrs,pi->iqrs', eri, P_imp)
    T = np.einsum('iqrs,qj->ijrs', T,   P_imp)
    T = np.einsum('ijrs,rk->ijks', T,   P_imp)
    eri_proj = np.einsum('ijks,sl->ijkl', T, P_imp)   # (n_imp, n_imp, n_imp, n_imp)

    # Fock embedding from frozen core (explicit np.ix_ blocks — avoids full 36^4 einsum)
    # v_fock[p,q] = Σ_{r,s∈env} gamma_core_env[r,s] * (eri[p,q,r,s] - 0.5*eri[p,r,s,q])
    env = np.array(env_orbs)
    gamma_env_blk = gamma_core_full[np.ix_(env, env)]                             # (n_env, n_env)
    arange        = np.arange(n_orb)
    eri_J = eri[np.ix_(arange, arange, env, env)]                                # (n_orb, n_orb, n_env, n_env)
    eri_K = eri[np.ix_(arange, env,    env, arange)]                             # (n_orb, n_env, n_env, n_orb)
    J_full = np.einsum('rs,pqrs->pq', gamma_env_blk, eri_J)                      # (n_orb, n_orb)
    K_full = np.einsum('rs,prsq->pq', gamma_env_blk, eri_K)                      # (n_orb, n_orb)
    v_fock_full = J_full - 0.5 * K_full
    v_fock_imp  = P_imp.T @ v_fock_full @ P_imp                                  # (n_imp, n_imp)

    h1_solver = h1_phys_proj + v_fock_imp

    return h1_phys_proj, h1_solver, eri_proj


def impurity_electron_count(
    n_elec_total: int,
    n_elec_core: float,
) -> tuple[int, int]:
    """
    Determine (n_alpha_imp, n_beta_imp) for the impurity TrimCI solve.

    Assumes closed-shell: n_alpha == n_beta. Raises AssertionError if
    n_elec_imp is odd (indicates a bad core electron count).

    Parameters
    ----------
    n_elec_total : total electrons in the full system (54 for Fe4S4)
    n_elec_core  : Tr(gamma_core) — electrons in the frozen core (float, from schmidt_decomp)

    Returns
    -------
    (n_alpha_imp, n_beta_imp) : both equal to n_elec_imp // 2
    """
    n_elec_imp = int(round(n_elec_total - n_elec_core))
    assert n_elec_imp % 2 == 0, (
        f"n_elec_imp = {n_elec_imp} is odd — check core electron count "
        f"(n_elec_total={n_elec_total}, n_elec_core={n_elec_core:.4f})")
    n_half = n_elec_imp // 2
    return n_half, n_half
```

- [ ] **Step 4: Run all bath tests**

```bash
python -m pytest TrimCI_Flow/tests/test_dmet_bath.py -v
```

Expected: all tests pass (no failures).

- [ ] **Step 5: Commit**

```bash
git add TrimCI_Flow/dmet/bath.py TrimCI_Flow/tests/test_dmet_bath.py
git commit -m "feat(dmet): build_impurity_hamiltonian + impurity_electron_count"
```

---

## Task 4: Energy formulas

**Files:**
- Create: `TrimCI_Flow/dmet/energy.py`
- Create: `TrimCI_Flow/tests/test_dmet_energy.py`

- [ ] **Step 1: Write failing tests**

```python
# TrimCI_Flow/tests/test_dmet_energy.py
import numpy as np
import pytest


def _setup_4orb_impurity():
    """Minimal 4-orbital impurity: gamma=I, h1_phys=I, h1_solver=2I, eri=zeros.
    Known results: A=2, B=2, C=2 (with nfrag=2)."""
    n_imp = 4
    nfrag = 2
    gamma  = np.eye(n_imp)
    gamma2 = np.zeros((n_imp,) * 4)   # zero 2-RDM → E_2b = 0 for all formulas
    h1_phys   = np.eye(n_imp)
    h1_solver = 2.0 * np.eye(n_imp)   # v_fock adds I → h1_solver = 2I
    eri_proj  = np.zeros((n_imp,) * 4)
    return h1_phys, h1_solver, eri_proj, gamma, gamma2, nfrag


def test_dmet_energy_a_identity():
    """A(h1_phys=I, gamma=I, nfrag=2): Σ_{p<2} h1[p,p]*gamma[p,p] = 1+1 = 2."""
    from TrimCI_Flow.dmet.energy import dmet_energy_a
    h1_phys, _, _, gamma, _, nfrag = _setup_4orb_impurity()
    result = dmet_energy_a(h1_phys, gamma, nfrag)
    assert np.isclose(result, 2.0), f"Expected 2.0, got {result}"


def test_dmet_energy_b_uses_h1_solver():
    """B uses h1_solver (2I), so 1-body part = Σ_{p<2} 2*gamma[p,p] = 4."""
    from TrimCI_Flow.dmet.energy import dmet_energy_b
    _, h1_solver, eri_proj, gamma, gamma2, nfrag = _setup_4orb_impurity()
    result = dmet_energy_b(h1_solver, eri_proj, gamma, gamma2, nfrag)
    # E_1b = Σ_{p<2, q} h1_solver[p,q]*gamma[p,q] = 2*1 + 2*1 = 4 (diagonal gamma)
    # E_2b = 0 (zero eri)
    assert np.isclose(result, 4.0), f"Expected 4.0, got {result}"


def test_dmet_energy_a_not_equal_b_when_fock_nonzero():
    """A and B differ when h1_solver != h1_phys_proj."""
    from TrimCI_Flow.dmet.energy import dmet_energy_a, dmet_energy_b
    h1_phys, h1_solver, eri_proj, gamma, gamma2, nfrag = _setup_4orb_impurity()
    a = dmet_energy_a(h1_phys, gamma, nfrag)
    b = dmet_energy_b(h1_solver, eri_proj, gamma, gamma2, nfrag)
    assert not np.isclose(a, b), "A and B should differ when h1_solver != h1_phys"


def test_dmet_energy_c_democratic_fraction():
    """C = full_impurity_energy * (nfrag / n_imp). With h1_solver=2I, gamma=I, eri=0:
    E_full = Σ_p 2*1 = 8; C = 8 * (2/4) = 4."""
    from TrimCI_Flow.dmet.energy import dmet_energy_c
    _, h1_solver, eri_proj, gamma, gamma2, nfrag = _setup_4orb_impurity()
    n_imp = h1_solver.shape[0]
    result = dmet_energy_c(h1_solver, eri_proj, gamma, gamma2, nfrag)
    expected = (np.einsum('pq,pq->', h1_solver, gamma)) * (nfrag / n_imp)
    assert np.isclose(result, expected), f"Expected {expected}, got {result}"


def test_dmet_energy_b_includes_2body():
    """With nonzero eri and gamma2, E_2b contributes to B."""
    from TrimCI_Flow.dmet.energy import dmet_energy_b
    n_imp, nfrag = 4, 2
    h1_solver = np.zeros((n_imp, n_imp))
    gamma     = np.zeros((n_imp, n_imp))
    # Construct a gamma2 with a known nonzero [0,1,0,1] element
    gamma2 = np.zeros((n_imp,) * 4)
    gamma2[0, 1, 0, 1] = 2.0
    eri_proj = np.zeros((n_imp,) * 4)
    eri_proj[0, 1, 0, 1] = 3.0    # p=0 ∈ frag → contributes to E_2b
    # E_2b = 0.5 * eri[0,1,0,1]*gamma2[0,1,0,1] = 0.5 * 3.0 * 2.0 = 3.0
    result = dmet_energy_b(h1_solver, eri_proj, gamma, gamma2, nfrag)
    assert np.isclose(result, 3.0), f"Expected 3.0, got {result}"


def test_dmet_energy_b_only_counts_fragment_rows():
    """eri[2,1,0,1] (p=2 NOT in frag) must NOT contribute to E_2b."""
    from TrimCI_Flow.dmet.energy import dmet_energy_b
    n_imp, nfrag = 4, 2
    h1_solver = np.zeros((n_imp, n_imp))
    gamma     = np.zeros((n_imp, n_imp))
    gamma2    = np.zeros((n_imp,) * 4)
    gamma2[2, 1, 2, 1] = 2.0
    eri_proj  = np.zeros((n_imp,) * 4)
    eri_proj[2, 1, 2, 1] = 3.0    # p=2 NOT in frag (nfrag=2) → excluded
    result = dmet_energy_b(h1_solver, eri_proj, gamma, gamma2, nfrag)
    assert np.isclose(result, 0.0), f"Expected 0.0, got {result}"


def test_check_2rdm_convention_function_exists():
    """Smoke test: check_2rdm_convention is importable."""
    from TrimCI_Flow.dmet.energy import check_2rdm_convention
    assert callable(check_2rdm_convention)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest TrimCI_Flow/tests/test_dmet_energy.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'TrimCI_Flow.dmet.energy'`

- [ ] **Step 3: Implement energy.py**

```python
# TrimCI_Flow/dmet/energy.py
"""
energy.py
=========
DMET fragment energy formulas for 1-shot non-overlapping embedding.

Three formulas are provided:
  A — 1-body only (no 2-RDM). DEBUG / baseline. Uses h1_phys_proj.
  B — 1-RDM + 2-RDM fragment partition. PRIMARY. Uses h1_solver.
  C — Democratic impurity energy. DIAGNOSTIC. Uses h1_solver.

The 2-RDM convention is gated by check_2rdm_convention, which must
pass before any B or C contraction is trusted.
"""
from __future__ import annotations

import numpy as np


def dmet_energy_a(
    h1_phys_proj: np.ndarray,
    gamma: np.ndarray,
    nfrag: int,
) -> float:
    """
    Approach A: 1-body only, physical h1. DEBUG / baseline.

    Uses h1_phys_proj (NO Fock/core correction). The A−B difference reveals
    both the missing 2e terms AND the frozen-core Fock correction — a double
    diagnostic for sanity-checking B.

    E_A = Σ_{p∈frag, q∈imp} h1_phys_proj[p,q] * γ[p,q]
    """
    return float(np.einsum('pq,pq->', h1_phys_proj[:nfrag, :], gamma[:nfrag, :]))


def dmet_energy_b(
    h1_solver: np.ndarray,
    eri_proj: np.ndarray,
    gamma: np.ndarray,
    gamma2: np.ndarray,
    nfrag: int,
) -> float:
    """
    Approach B: 1-RDM + 2-RDM fragment partition. PRIMARY / DEFAULT.

    E_I = Σ_{p∈frag, q∈imp} h1_solver[p,q] * γ[p,q]
        + (1/2) Σ_{p∈frag, q,r,s∈imp} eri_proj[p,q,r,s] * Γ2[p,q,r,s]

    h1_solver = h1_phys_proj + v_fock_core_imp. The Fock correction captures
    the mean-field interaction between fragment electrons and the frozen core
    density. Without it, E_DMET_B would omit the fragment–core Coulomb/exchange
    contribution entirely.

    eri convention: chemist notation (pq|rs).
    gamma2 convention: verified via check_2rdm_convention before calling this.
    """
    E_1b = float(np.einsum('pq,pq->', h1_solver[:nfrag, :], gamma[:nfrag, :]))
    E_2b = 0.5 * float(np.einsum('pqrs,pqrs->', eri_proj[:nfrag, :, :, :], gamma2[:nfrag, :, :, :]))
    return E_1b + E_2b


def dmet_energy_c(
    h1_solver: np.ndarray,
    eri_proj: np.ndarray,
    gamma: np.ndarray,
    gamma2: np.ndarray,
    nfrag: int,
) -> float:
    """
    Approach C: democratic impurity energy. DIAGNOSTIC only.

    Assigns fraction (nfrag / n_imp) of the full impurity energy to the fragment.
    Uses h1_solver so the frozen-core correction is on the same footing as B.
    No guaranteed ordering relative to A or B — useful for bracketing sanity check.
    """
    n_imp = h1_solver.shape[0]
    E_1b  = float(np.einsum('pq,pq->', h1_solver, gamma))
    E_2b  = 0.5 * float(np.einsum('pqrs,pqrs->', eri_proj, gamma2))
    return (E_1b + E_2b) * (nfrag / n_imp)


def check_2rdm_convention(
    dets,
    coeffs,
    n_imp: int,
    h1_solver: np.ndarray,
    eri_proj: np.ndarray,
    e_trimci: float,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Verify that energy_from_rdm(gamma, gamma2, h1_solver, eri_proj, 0.0, n_imp)
    reproduces TrimCI's reported energy within `tol`. Called on fragment 0 only.

    Raises RuntimeError if the discrepancy exceeds tol. This gates ALL Approach B
    contractions — if the check fails, no B or C energies are reported.

    Parameters
    ----------
    dets      : TrimCI Determinant C++ objects (from FragmentResult.dets)
    coeffs    : CI coefficients (from FragmentResult.coeffs)
    n_imp     : impurity orbital count (24 for Fe4S4 DMET)
    h1_solver : (n_imp, n_imp) impurity h1 including Fock correction
    eri_proj  : (n_imp, n_imp, n_imp, n_imp) projected ERI
    e_trimci  : TrimCI variational energy for this fragment (from result.energy)
    tol       : tolerance in Ha (default 1e-6)

    Returns
    -------
    (gamma, gamma2) : (n_imp, n_imp) and (n_imp, n_imp, n_imp, n_imp) arrays,
                      ready for energy formulas B and C.
    """
    from trimci.trimci_core import compute_1rdm, compute_2rdm, energy_from_rdm

    gamma  = np.asarray(compute_1rdm(dets, list(coeffs), n_imp)).reshape(n_imp, n_imp)
    gamma2 = np.asarray(compute_2rdm(dets, list(coeffs), n_imp)).reshape(
                 n_imp, n_imp, n_imp, n_imp)

    # energy_from_rdm binding:
    #   gamma   → flat list
    #   gamma2  → flat list
    #   h1      → 2D list (std::vector<std::vector<double>>)
    #   eri     → flat list
    e_check = energy_from_rdm(
        gamma.ravel().tolist(),
        gamma2.ravel().tolist(),
        h1_solver.tolist(),        # 2D list — NOT .ravel()
        eri_proj.ravel().tolist(),
        0.0,
        n_imp,
    )
    discrepancy = abs(float(e_check) - e_trimci)
    if discrepancy > tol:
        raise RuntimeError(
            f"2-RDM convention check FAILED on fragment 0: "
            f"energy_from_rdm={float(e_check):.8f} Ha, "
            f"TrimCI reported={e_trimci:.8f} Ha, "
            f"discrepancy={discrepancy:.2e} Ha (tol={tol:.0e}). "
            f"Check h1_solver (must include Fock correction) and eri_proj convention."
        )
    print(f"  [DMET] 2-RDM convention check PASSED (discrepancy={discrepancy:.2e} Ha)")
    return gamma, gamma2
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest TrimCI_Flow/tests/test_dmet_energy.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add TrimCI_Flow/dmet/energy.py TrimCI_Flow/tests/test_dmet_energy.py
git commit -m "feat(dmet): energy formulas A/B/C + check_2rdm_convention gate"
```

---

## Task 5: Solver orchestration

**Files:**
- Create: `TrimCI_Flow/dmet/solver.py`

No unit tests for solver.py — it orchestrates TrimCI C++ objects that cannot be
mocked without the binary. Integration-tested in Task 8.

- [ ] **Step 1: Implement solver.py**

```python
# TrimCI_Flow/dmet/solver.py
"""
solver.py
=========
1-shot non-overlapping DMET orchestration.

Fragment partition: 3 non-overlapping groups of n_orb//3 orbitals,
sorted by h1 diagonal energy (same ordering as Phase C / meanfield).

Pipeline per fragment:
  1. schmidt_decomp(gamma_mf, frag_orbs, n_orb)
  2. build_impurity_hamiltonian(h1, eri, gamma_core_full, env_orbs, P_imp)
  3. impurity_electron_count(n_elec_total, n_elec_core)
  4. solve_fragment_trimci(h1_solver, eri_proj, n_alpha_imp, n_beta_imp, n_imp)
  5. check_2rdm_convention(...)  — fragment 0 only
  6. compute_1rdm / compute_2rdm for fragments 1+
  7. dmet_energy_a/b/c per fragment

E_DMET = E_nuc + sum(E_I_b for I in fragments)
"""
from __future__ import annotations

import os
import json
import numpy as np
from typing import Optional

from TrimCI_Flow.core.results import FragmentedRunResult
from TrimCI_Flow.core.fragment import (
    fragment_by_sliding_window,
    extract_fragment_integrals,
    fragment_electron_count,
)
from TrimCI_Flow.core.trimci_adapter import solve_fragment_trimci
from TrimCI_Flow.dmet.hf_reference import run_hf
from TrimCI_Flow.dmet.bath import (
    schmidt_decomp,
    build_impurity_hamiltonian,
    impurity_electron_count,
)
from TrimCI_Flow.dmet.energy import (
    check_2rdm_convention,
    dmet_energy_a,
    dmet_energy_b,
    dmet_energy_c,
)
from trimci.trimci_core import compute_1rdm, compute_2rdm


def run_dmet_1shot(
    fcidump_path: str,
    trimci_config: Optional[dict] = None,
    output_dir: Optional[str] = None,
) -> FragmentedRunResult:
    """
    1-shot non-overlapping DMET with TrimCI as impurity solver.

    Parameters
    ----------
    fcidump_path  : path to FCIDUMP file (9.6 MB Fe4S4 file)
    trimci_config : optional TrimCI config overrides (passed to solve_fragment_trimci)
    output_dir    : if provided, writes results.json here

    Returns
    -------
    FragmentedRunResult with runtime attributes:
      .E_dmet        : float, canonical DMET energy from formula B (Ha)
      .E_dmet_a      : float, formula A energy (debug)
      .E_dmet_c      : float, formula C energy (diagnostic)
      .E_hf          : float, RHF baseline energy
      .hf_converged  : bool
    """
    import trimci as _trimci

    # ── 1. Read FCIDUMP ───────────────────────────────────────────────────────
    h1, eri, n_elec, n_orb, E_nuc, n_alpha, n_beta, psym = _trimci.read_fcidump(fcidump_path)
    print(f"  [DMET] FCIDUMP: n_orb={n_orb}, n_elec={n_elec}, E_nuc={E_nuc}")

    # ── 2. Fragment partition (non-overlapping, h1-diagonal ordering) ─────────
    order = np.argsort(np.diag(h1))       # same ordering as Phase C/B
    n_frag = n_orb // 3                   # = 12 for Fe4S4
    fragments = [
        sorted(order[0        : n_frag].tolist()),
        sorted(order[n_frag   : 2*n_frag].tolist()),
        sorted(order[2*n_frag : n_orb].tolist()),
    ]
    print(f"  [DMET] Fragments: F0={fragments[0][:3]}..., F1={fragments[1][:3]}..., F2={fragments[2][:3]}...")

    # ── 3. RHF reference ─────────────────────────────────────────────────────
    gamma_mf, e_hf = run_hf(h1, eri, n_elec, n_orb)
    print(f"  [DMET] E_HF = {e_hf:.6f} Ha  (Tr(gamma_mf)={np.trace(gamma_mf):.4f})")

    # ── 4. Per-fragment impurity solve ────────────────────────────────────────
    fragment_energies   = []
    fragment_n_dets     = []
    energies_a          = []
    energies_b          = []
    energies_c          = []
    convention_checked  = False

    for frag_idx, frag_orbs in enumerate(fragments):
        print(f"\n  [DMET] === Fragment {frag_idx} ({len(frag_orbs)} orbs) ===")

        # Schmidt decomp + bath
        P_imp, gamma_core_full, n_elec_core, n_bath, env_orbs = schmidt_decomp(
            gamma_mf, frag_orbs, n_orb)
        n_imp = len(frag_orbs) + n_bath
        print(f"    n_bath={n_bath}, n_core={len(env_orbs)-n_bath}, n_elec_core={n_elec_core:.3f}")

        # Sanity: electron conservation
        n_alpha_imp, n_beta_imp = impurity_electron_count(n_elec, n_elec_core)
        print(f"    n_elec_imp={n_alpha_imp+n_beta_imp} (alpha={n_alpha_imp}, beta={n_beta_imp})")

        # Impurity Hamiltonian
        h1_phys_proj, h1_solver, eri_proj = build_impurity_hamiltonian(
            h1, eri, gamma_core_full, env_orbs, P_imp)

        # TrimCI solve
        result = solve_fragment_trimci(
            h1_solver, eri_proj,
            n_alpha_imp, n_beta_imp,
            n_orb_frag=n_imp,
            config=trimci_config,
        )
        print(f"    E_imp={result.energy:.6f} Ha, n_dets={result.n_dets}")
        fragment_energies.append(result.energy)
        fragment_n_dets.append(result.n_dets)

        # RDM extraction + 2-RDM convention check (fragment 0 only)
        if not convention_checked:
            gamma, gamma2 = check_2rdm_convention(
                result.dets, result.coeffs, n_imp,
                h1_solver, eri_proj, result.energy)
            convention_checked = True
        else:
            gamma  = np.asarray(compute_1rdm(result.dets, list(result.coeffs), n_imp)).reshape(n_imp, n_imp)
            gamma2 = np.asarray(compute_2rdm(result.dets, list(result.coeffs), n_imp)).reshape(
                         n_imp, n_imp, n_imp, n_imp)

        # Energy formulas
        nfrag = len(frag_orbs)
        ea = dmet_energy_a(h1_phys_proj, gamma, nfrag)
        eb = dmet_energy_b(h1_solver,    eri_proj, gamma, gamma2, nfrag)
        ec = dmet_energy_c(h1_solver,    eri_proj, gamma, gamma2, nfrag)
        print(f"    E_A={ea:.6f}  E_B={eb:.6f}  E_C={ec:.6f}")
        energies_a.append(ea); energies_b.append(eb); energies_c.append(ec)

    # ── 5. Total energies ─────────────────────────────────────────────────────
    E_dmet_a = E_nuc + sum(energies_a)
    E_dmet_b = E_nuc + sum(energies_b)
    E_dmet_c = E_nuc + sum(energies_c)

    print(f"\n{'='*60}")
    print(f"=== DMET 1-shot (non-overlapping 12+12+12) ===")
    print(f"  E_HF                           = {e_hf:.6f} Ha")
    print(f"  E_DMET_A (1-body, debug)       = {E_dmet_a:.6f} Ha")
    print(f"  E_DMET_B (2-RDM, primary)      = {E_dmet_b:.6f} Ha  ← canonical")
    print(f"  E_DMET_C (democratic, diag.)   = {E_dmet_c:.6f} Ha")
    print(f"  Reference (brute-force TrimCI) = -327.1920 Ha")
    print(f"  Error (B - reference)          = {E_dmet_b - (-327.1920):+.4f} Ha")
    print(f"  Total DMET dets: {sum(fragment_n_dets)}  (Phase C baseline: 118,  brute-force: 10095)")
    print(f"{'='*60}")

    # ── 6. Build result object ────────────────────────────────────────────────
    result_obj = FragmentedRunResult(
        fragment_energies = fragment_energies,
        fragment_n_dets   = fragment_n_dets,
        fragment_orbs     = fragments,
        total_dets        = sum(fragment_n_dets),
        iterations        = 1,
    )
    result_obj.E_dmet       = E_dmet_b
    result_obj.E_dmet_a     = E_dmet_a
    result_obj.E_dmet_c     = E_dmet_c
    result_obj.E_hf         = e_hf

    # ── 7. Optionally write JSON ──────────────────────────────────────────────
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        payload = {
            "E_dmet_b":        E_dmet_b,
            "E_dmet_a":        E_dmet_a,
            "E_dmet_c":        E_dmet_c,
            "E_hf":            e_hf,
            "fragment_n_dets": fragment_n_dets,
            "total_dets":      sum(fragment_n_dets),
            "fragment_orbs":   fragments,
            "fragment_energies_imp": fragment_energies,
        }
        with open(os.path.join(output_dir, "results.json"), "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  [DMET] Results written to {output_dir}/results.json")

    return result_obj
```

- [ ] **Step 2: Verify solver imports cleanly**

```bash
python -c "from TrimCI_Flow.dmet.solver import run_dmet_1shot; print('import OK')"
```

Expected: `import OK`

- [ ] **Step 3: Commit**

```bash
git add TrimCI_Flow/dmet/solver.py
git commit -m "feat(dmet): run_dmet_1shot orchestration"
```

---

## Task 6: Package init + runner script

**Files:**
- Modify: `TrimCI_Flow/dmet/__init__.py`
- Create: `TrimCI_Flow/dmet/runners/__init__.py`
- Create: `TrimCI_Flow/dmet/runners/run_dmet_1shot.py`

- [ ] **Step 1: Update dmet/__init__.py**

Read the current content first:

```bash
cat /home/unfunnypanda/Proj_Flow/TrimCI_Flow/dmet/__init__.py
```

Replace the entire file with:

```python
# TrimCI_Flow/dmet/__init__.py
"""
TrimCI_Flow.dmet — 1-shot non-overlapping DMET with TrimCI impurity solver.
Path A in the original TrimCI-Flow plan.
"""
from TrimCI_Flow.dmet.solver import run_dmet_1shot

__all__ = ["run_dmet_1shot"]
```

- [ ] **Step 2: Create runners/__init__.py**

```bash
touch /home/unfunnypanda/Proj_Flow/TrimCI_Flow/dmet/runners/__init__.py
```

- [ ] **Step 3: Create the runner script**

```python
# TrimCI_Flow/dmet/runners/run_dmet_1shot.py
"""
Runner: 1-shot non-overlapping DMET on Fe4S4.

Usage (from /home/unfunnypanda/Proj_Flow/):
    source qflowenv/bin/activate
    python TrimCI_Flow/dmet/runners/run_dmet_1shot.py 2>&1 | tee output.log
"""
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

FCIDUMP = (
    "/home/unfunnypanda/Proj_Flow/"
    "Fe4S4_251230orbital_-327.1920_10kdets/"
    "Fe4S4_251230orbital_-327.1920_10kdets/"
    "fcidump_cycle_6"
)

TRIMCI_CONFIG = {
    "threshold":        0.06,
    "max_final_dets":   "auto",   # ≈51 for 24-orbital impurity
    "max_rounds":       2,
    "num_runs":         1,
    "pool_build_strategy": "heat_bath",
    "verbose":          False,
}

if __name__ == "__main__":
    from TrimCI_Flow.dmet import run_dmet_1shot

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(
        os.path.dirname(__file__), '..', '..', 'Outputs',
        f"outs_dmet_1shot_{timestamp}")
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"[runner] Output dir: {output_dir}")
    result = run_dmet_1shot(
        fcidump_path = FCIDUMP,
        trimci_config = TRIMCI_CONFIG,
        output_dir    = output_dir,
    )
    print(f"[runner] Done. E_DMET = {result.E_dmet:.6f} Ha")
```

- [ ] **Step 4: Smoke test the import chain**

```bash
python -c "from TrimCI_Flow.dmet import run_dmet_1shot; print('OK')"
```

Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add TrimCI_Flow/dmet/__init__.py TrimCI_Flow/dmet/runners/__init__.py TrimCI_Flow/dmet/runners/run_dmet_1shot.py
git commit -m "feat(dmet): package init + run_dmet_1shot runner"
```

---

## Task 7: Phase C regression

**Files:** No changes.

- [ ] **Step 1: Run the Phase C regression check**

```bash
python - <<'EOF'
from TrimCI_Flow.uncoupled.solver import run_fragmented_trimci

FCIDUMP = (
    "/home/unfunnypanda/Proj_Flow/"
    "Fe4S4_251230orbital_-327.1920_10kdets/"
    "Fe4S4_251230orbital_-327.1920_10kdets/"
    "fcidump_cycle_6"
)
result = run_fragmented_trimci(FCIDUMP)
assert result.total_dets == 118,          f"FAIL: total_dets={result.total_dets}, expected 118"
assert result.fragment_n_dets == [51, 51, 16], f"FAIL: {result.fragment_n_dets}"
print(f"Phase C regression PASSED: total_dets={result.total_dets}, fragment_n_dets={result.fragment_n_dets}")
EOF
```

Expected output: `Phase C regression PASSED: total_dets=118, fragment_n_dets=[51, 51, 16]`

If this fails, something in the new DMET code has accidentally modified `uncoupled/` — check the import chain immediately.

---

## Task 8: Full integration run

**Files:**
- Modify: `TrimCI_Flow/progress.md` (append)

- [ ] **Step 1: Run the DMET integration**

```bash
cd /home/unfunnypanda/Proj_Flow
source qflowenv/bin/activate
python TrimCI_Flow/dmet/runners/run_dmet_1shot.py 2>&1 | tee /tmp/dmet_run.log
```

**Expected output structure:**

```
[runner] Output dir: .../Outputs/outs_dmet_1shot_YYYYMMDD_HHMMSS
  [DMET] FCIDUMP: n_orb=36, n_elec=54, E_nuc=0.0
  [DMET] Fragments: F0=[...], F1=[...], F2=[...]
  [DMET] E_HF = -XXX.XXXXXX Ha  (Tr(gamma_mf)=54.0000)
  [DMET] === Fragment 0 (12 orbs) ===
    n_bath=12, n_core=12, n_elec_core=XX.XXX
    n_elec_imp=XX (alpha=XX, beta=XX)
    E_imp=...  n_dets=...
  [DMET] 2-RDM convention check PASSED (discrepancy=X.XXe-XX Ha)
  [DMET] === Fragment 1 (12 orbs) ===
    ...
  [DMET] === Fragment 2 (12 orbs) ===
    ...
============================================================
=== DMET 1-shot (non-overlapping 12+12+12) ===
  E_HF                           = -XXX.XXXXXX Ha
  E_DMET_A (1-body, debug)       = -XXX.XXXXXX Ha
  E_DMET_B (2-RDM, primary)      = -XXX.XXXXXX Ha  ← canonical
  E_DMET_C (democratic, diag.)   = -XXX.XXXXXX Ha
  Reference (brute-force TrimCI) = -327.1920 Ha
  Error (B - reference)          = +X.XXXX Ha
  Total DMET dets: XXX  (Phase C baseline: 118,  brute-force: 10095)
============================================================
```

**Failure modes to diagnose:**

| Symptom | Likely cause |
|---------|-------------|
| `2-RDM convention check FAILED` | `energy_from_rdm` h1 argument shape wrong; check `.tolist()` not `.ravel().tolist()` for h1 |
| `AssertionError: n_elec_imp=XX is odd` | `n_elec_core` from core trace is not close to integer; check gamma_mf from RHF |
| `RHF WARNING` + bad gamma_mf | Fe4S4 RHF unlikely to converge — expected, continue; bath construction still valid |
| TrimCI segfault on 24-orb problem | `n_alpha_imp` or `n_beta_imp` miscounted; print and verify before TrimCI call |

- [ ] **Step 2: Verify the JSON output exists and is valid**

```bash
ls -la /home/unfunnypanda/Proj_Flow/TrimCI_Flow/Outputs/outs_dmet_1shot_*/results.json
python -c "import json; d=json.load(open('$(ls /home/unfunnypanda/Proj_Flow/TrimCI_Flow/Outputs/outs_dmet_1shot_*/results.json | head -1)')); print(list(d.keys()))"
```

Expected keys: `['E_dmet_b', 'E_dmet_a', 'E_dmet_c', 'E_hf', 'fragment_n_dets', 'total_dets', 'fragment_orbs', 'fragment_energies_imp']`

- [ ] **Step 3: Append DMET run entry to progress.md**

Append the following section (fill in actual values from the run output):

```markdown
# DMET — Run 1 (2026-04-15): 1-shot non-overlapping DMET

## Status: [SUCCESS / PARTIAL / FAILURE]

## Configuration
```
Method: 1-shot DMET, non-overlapping 12+12+12 fragments
HF reference: PySCF RHF (converged: True/False)
Fragment partition: h1-diagonal ordering, [F0: orbs 0-11, F1: 12-23, F2: 24-35 in energy rank]
Impurity size: 24 orbitals per fragment (12 frag + 12 bath)
TrimCI config: threshold=0.06, max_final_dets="auto"
Output: TrimCI_Flow/Outputs/outs_dmet_1shot_TIMESTAMP/
```

## Results
```
E_HF                = [value] Ha
E_DMET_A (debug)    = [value] Ha
E_DMET_B (primary)  = [value] Ha
E_DMET_C (diag.)    = [value] Ha
Reference           = -327.1920 Ha
Error (B-ref)       = [value] Ha

Fragment dets: [n_dets_F0, n_dets_F1, n_dets_F2]
Total DMET dets: [value]
Phase C baseline:   118
Brute-force:        10,095
2-RDM convention check: PASSED / FAILED
```

## Functions added or changed
- NEW: TrimCI_Flow/dmet/hf_reference.py — run_hf()
- NEW: TrimCI_Flow/dmet/bath.py — schmidt_decomp(), build_impurity_hamiltonian(), impurity_electron_count()
- NEW: TrimCI_Flow/dmet/energy.py — check_2rdm_convention(), dmet_energy_a(), dmet_energy_b(), dmet_energy_c()
- NEW: TrimCI_Flow/dmet/solver.py — run_dmet_1shot()
- NEW: TrimCI_Flow/dmet/runners/run_dmet_1shot.py
- MOD: TrimCI_Flow/dmet/__init__.py — export run_dmet_1shot

## Notes
[Observations about RHF convergence, E_DMET_B value vs reference, det counts]
```

- [ ] **Step 4: Final commit**

```bash
git add TrimCI_Flow/progress.md
git commit -m "feat(dmet): DMET v1 1-shot run — E_DMET_B=[value] Ha, [N] total dets"
```

---

## Self-review checklist

- [ ] Spec §4.1 (fragment partition): implemented in `solver.py` lines that build `fragments`
- [ ] Spec §4.2 (HF reference): `hf_reference.py::run_hf` — warns but does not raise on non-convergence
- [ ] Spec §4.3 (Schmidt decomp): `bath.py::schmidt_decomp` — returns 5-tuple including env_orbs
- [ ] Spec §4.3 (impurity Hamiltonian): `bath.py::build_impurity_hamiltonian` — np.ix_ Fock option implemented
- [ ] Spec §4.4 (electron count): `bath.py::impurity_electron_count` — asserts on odd n_elec_imp
- [ ] Spec §4.5 (TrimCI solve): uses `solve_fragment_trimci` unchanged from `core/trimci_adapter.py`
- [ ] Spec §4.6 (convention check): runs on fragment 0 only; gate raises RuntimeError on failure
- [ ] Spec §4.7 (energy formulas): A uses h1_phys_proj; B and C use h1_solver
- [ ] Spec §5 (output): all four energies + det counts printed and written to JSON
- [ ] Spec §6 test 1 (convention check): gated in `check_2rdm_convention`
- [ ] Spec §6 test 2 (electron conservation): `impurity_electron_count` asserts per fragment
- [ ] Spec §6 test 3 (orthogonality): validated by `test_p_imp_columns_orthonormal_*`
- [ ] Spec §6 test 4 (Phase C regression): Task 7
- [ ] Spec §6 test 5 (all three logged): solver.py prints A, B, C for each fragment
- [ ] `energy_from_rdm` h1 argument: `.tolist()` (2D), not `.ravel().tolist()`
- [ ] No modifications to `uncoupled/`, `meanfield/`, `core/`
