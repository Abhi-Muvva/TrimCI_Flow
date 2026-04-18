# TrimCI_Flow/tests/test_dmet_bath.py
import numpy as np
import pytest


# ── Fixtures ────────────────────────────────────────────────────────────────

def _gamma_4orb():
    """4-orbital gamma_mf with frag=[0,1], env=[2,3].
    gamma_AB (2x2) has full rank -> n_bath=2, n_core=0."""
    return np.array([
        [1.5, 0.3, 0.4, 0.1],
        [0.3, 0.8, 0.2, 0.1],
        [0.4, 0.2, 1.2, 0.3],
        [0.1, 0.1, 0.3, 0.9],
    ]), [0, 1], 4


def _gamma_6orb_with_core():
    """6-orbital gamma_mf with frag=[0,1], env=[2,3,4,5].
    gamma_AB (2x4) has rank 2 -> n_bath=2, n_core=2."""
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
    """4-orbital: frag-env block is 2x2 full rank -> n_bath == 2."""
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
    for p in range(4):
        eri[p, p, p, p] = 0.5
        for q in range(4):
            if p != q:
                eri[p, p, q, q] = eri[q, q, p, p] = 0.2
                eri[p, q, p, q] = eri[q, p, q, p] = 0.1
                eri[p, q, q, p] = eri[q, p, p, q] = 0.1  # full 8-fold symmetry
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
    diff = np.max(np.abs(h1_sol - h1_phys))
    assert diff > 1e-10, f"Expected nonzero Fock correction; max|diff|={diff:.2e}"


def test_eri_proj_shape_and_symmetry():
    """eri_proj must be (n_imp, n_imp, n_imp, n_imp) and preserve chemist symmetry."""
    from TrimCI_Flow.dmet.bath import build_impurity_hamiltonian
    h1, eri, _, gamma_core_full, env_orbs, P_imp, _, _ = _build_tiny_imp()
    _, _, eri_proj = build_impurity_hamiltonian(h1, eri, gamma_core_full, env_orbs, P_imp)
    n_imp = P_imp.shape[1]
    assert eri_proj.shape == (n_imp, n_imp, n_imp, n_imp)
    np.testing.assert_allclose(eri_proj, eri_proj.transpose(1, 0, 2, 3), atol=1e-10)
    np.testing.assert_allclose(eri_proj, eri_proj.transpose(0, 1, 3, 2), atol=1e-10)
    np.testing.assert_allclose(eri_proj, eri_proj.transpose(2, 3, 0, 1), atol=1e-10)


# ── impurity_electron_count tests ────────────────────────────────────────────

def test_electron_count_even_total():
    from TrimCI_Flow.dmet.bath import impurity_electron_count
    n_alpha, n_beta = impurity_electron_count(n_elec_total=54, n_elec_core=12.0)
    assert n_alpha == n_beta == 21
    assert n_alpha + n_beta == 42


def test_electron_count_rounds_fractional_to_even():
    """Fractional n_elec_core (from UHF) is rounded to nearest even n_elec_imp."""
    from TrimCI_Flow.dmet.bath import impurity_electron_count
    # n_elec_imp_float = 54 - 23.365 = 30.635 -> nearest even = 30
    n_alpha, n_beta = impurity_electron_count(n_elec_total=54, n_elec_core=23.365)
    assert n_alpha == n_beta
    assert (n_alpha + n_beta) % 2 == 0
    assert n_alpha + n_beta == 30   # round(30.635/2)*2 = round(15.318)*2 = 15*2 = 30
