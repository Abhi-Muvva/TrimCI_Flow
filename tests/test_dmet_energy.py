# TrimCI_Flow/tests/test_dmet_energy.py
import numpy as np
import pytest


def _setup_4orb_impurity():
    """Minimal 4-orbital impurity: gamma=I, h1_phys=I, h1_solver=2I, eri=zeros.
    Known results: A=2, B=4, C=4 (with nfrag=2)."""
    n_imp = 4
    nfrag = 2
    gamma  = np.eye(n_imp)
    gamma2 = np.zeros((n_imp,) * 4)   # zero 2-RDM -> E_2b = 0 for all formulas
    h1_phys   = np.eye(n_imp)
    h1_solver = 2.0 * np.eye(n_imp)   # v_fock adds I -> h1_solver = 2I
    eri_proj  = np.zeros((n_imp,) * 4)
    return h1_phys, h1_solver, eri_proj, gamma, gamma2, nfrag


def test_dmet_energy_a_identity():
    """A(h1_phys=I, gamma=I, nfrag=2): sum_{p<2} h1[p,p]*gamma[p,p] = 1+1 = 2."""
    from TrimCI_Flow.dmet.energy import dmet_energy_a
    h1_phys, _, _, gamma, _, nfrag = _setup_4orb_impurity()
    result = dmet_energy_a(h1_phys, gamma, nfrag)
    assert np.isclose(result, 2.0), f"Expected 2.0, got {result}"


def test_dmet_energy_b_uses_h1_solver():
    """B uses h1_solver (2I), so 1-body part = sum_{p<2} 2*gamma[p,p] = 4."""
    from TrimCI_Flow.dmet.energy import dmet_energy_b
    _, h1_solver, eri_proj, gamma, gamma2, nfrag = _setup_4orb_impurity()
    result = dmet_energy_b(h1_solver, eri_proj, gamma, gamma2, nfrag)
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
    E_full = sum_p 2*1 = 8; C = 8 * (2/4) = 4."""
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
    gamma2 = np.zeros((n_imp,) * 4)
    gamma2[0, 1, 0, 1] = 2.0
    eri_proj = np.zeros((n_imp,) * 4)
    # physicist ERI: energy_from_rdm contracts eri[p,q,r,s]*(pr|qs).
    # For dmet_energy_b we use einsum 'prqs,pqrs->': eri_chem[p,r,q,s]*gamma2[p,q,r,s].
    # With gamma2[0,1,0,1]=2: need eri_chem[p=0, r=0, q=1, s=1] = 3.0.
    eri_proj[0, 0, 1, 1] = 3.0    # eri_chem[p,r,q,s] with p=0,r=0,q=1,s=1
    # E_2b = 0.5 * eri_chem[0,0,1,1]*gamma2[0,1,0,1] = 0.5 * 3.0 * 2.0 = 3.0
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
    eri_proj[2, 1, 2, 1] = 3.0    # p=2 NOT in frag (nfrag=2) -> excluded
    result = dmet_energy_b(h1_solver, eri_proj, gamma, gamma2, nfrag)
    assert np.isclose(result, 0.0), f"Expected 0.0, got {result}"


def test_check_2rdm_convention_function_exists():
    """Smoke test: check_2rdm_convention is importable."""
    from TrimCI_Flow.dmet.energy import check_2rdm_convention
    assert callable(check_2rdm_convention)
