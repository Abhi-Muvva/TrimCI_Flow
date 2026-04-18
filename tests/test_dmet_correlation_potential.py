# TrimCI_Flow/tests/test_dmet_correlation_potential.py
"""
Unit tests for dmet/correlation_potential.py.

All tests are pure numpy — no TrimCI or PySCF required.
"""
import numpy as np
import pytest

from TrimCI_Flow.dmet.correlation_potential import (
    build_u_emb,
    extract_gamma_frag,
    update_u_blocks,
    damp_u_blocks,
    gamma_frag_mismatch,
    u_blocks_max_change,
    zero_u_blocks,
)


# ── build_u_emb ──────────────────────────────────────────────────────────────

def test_build_u_emb_zero_blocks():
    """All-zero u_blocks → zero u_emb."""
    u_blocks  = [np.zeros((2, 2)), np.zeros((2, 2))]
    fragments = [[0, 1], [2, 3]]
    u_emb = build_u_emb(u_blocks, fragments, 4)
    np.testing.assert_allclose(u_emb, np.zeros((4, 4)))


def test_build_u_emb_block_diagonal_contiguous():
    """Contiguous fragments: u_emb must be exactly block-diagonal."""
    u0 = np.array([[1.0, 0.5], [0.5, 2.0]])
    u1 = np.array([[3.0, 0.1], [0.1, 4.0]])
    u_emb = build_u_emb([u0, u1], [[0, 1], [2, 3]], 4)
    np.testing.assert_allclose(u_emb[np.ix_([0, 1], [0, 1])], u0, atol=1e-12)
    np.testing.assert_allclose(u_emb[np.ix_([2, 3], [2, 3])], u1, atol=1e-12)
    # Off-diagonal blocks must be zero
    np.testing.assert_allclose(u_emb[np.ix_([0, 1], [2, 3])], np.zeros((2, 2)), atol=1e-12)
    np.testing.assert_allclose(u_emb[np.ix_([2, 3], [0, 1])], np.zeros((2, 2)), atol=1e-12)


def test_build_u_emb_non_contiguous_orbs():
    """Non-contiguous fragment orbitals are placed correctly in u_emb."""
    u0 = np.array([[1.0, 0.3], [0.3, 2.0]])
    u_emb = build_u_emb([u0], [[0, 3]], 5)
    assert u_emb[0, 0] == pytest.approx(1.0)
    assert u_emb[3, 3] == pytest.approx(2.0)
    assert u_emb[0, 3] == pytest.approx(0.3)
    assert u_emb[3, 0] == pytest.approx(0.3)
    # Untouched orbitals
    assert u_emb[1, 1] == pytest.approx(0.0)
    assert u_emb[2, 2] == pytest.approx(0.0)
    assert u_emb[4, 4] == pytest.approx(0.0)


def test_build_u_emb_shape():
    u_blocks  = [np.eye(3), np.eye(3), np.eye(3)]
    fragments = [list(range(0, 3)), list(range(3, 6)), list(range(6, 9))]
    u_emb = build_u_emb(u_blocks, fragments, 9)
    assert u_emb.shape == (9, 9)


def test_build_u_emb_is_symmetric():
    """u_emb should be symmetric when all u_I are symmetric."""
    u0 = np.array([[1.0, 0.4], [0.4, 2.0]])
    u1 = np.array([[3.0, 0.7], [0.7, 1.5]])
    u_emb = build_u_emb([u0, u1], [[0, 1], [2, 3]], 4)
    np.testing.assert_allclose(u_emb, u_emb.T, atol=1e-12)


# ── extract_gamma_frag ───────────────────────────────────────────────────────

def test_extract_gamma_frag_contiguous():
    gamma = np.arange(16.0).reshape(4, 4)
    result = extract_gamma_frag(gamma, [0, 1])
    expected = gamma[:2, :2]
    np.testing.assert_allclose(result, expected)


def test_extract_gamma_frag_non_contiguous():
    gamma = np.arange(25.0).reshape(5, 5)
    result = extract_gamma_frag(gamma, [1, 3])
    expected = gamma[np.ix_([1, 3], [1, 3])]
    np.testing.assert_allclose(result, expected)


def test_extract_gamma_frag_single_orbital():
    gamma = np.eye(4) * 1.5
    result = extract_gamma_frag(gamma, [2])
    np.testing.assert_allclose(result, np.array([[1.5]]))


# ── update_u_blocks ──────────────────────────────────────────────────────────

def test_update_u_blocks_increases_where_mf_exceeds_imp():
    """
    u must INCREASE where γ_MF > γ_imp.
    Raising u raises the orbital energy → UHF puts fewer electrons there
    → γ_MF[p,p] decreases toward γ_imp[p,p].  Sign convention: δu = γ_MF - γ_imp.
    """
    u_blocks  = [np.zeros((2, 2))]
    fragments = [[0, 1]]
    gamma_mf  = np.diag([1.5, 1.8, 0.8, 1.2])   # γ_MF[0,0]=1.5, γ_MF[1,1]=1.8
    gamma_imp = np.array([[1.0, 0.0], [0.0, 0.5]])  # both < γ_MF → u should increase
    new_u = update_u_blocks(u_blocks, gamma_mf, [gamma_imp], fragments, step=1.0)
    assert new_u[0][0, 0] > 0.0, "u[0,0] should increase when γ_MF > γ_imp"
    assert new_u[0][1, 1] > 0.0, "u[1,1] should increase when γ_MF > γ_imp"


def test_update_u_blocks_decreases_where_imp_exceeds_mf():
    """
    u must DECREASE where γ_imp > γ_MF.
    Lowering u lowers the orbital energy → UHF puts more electrons there
    → γ_MF[p,p] increases toward γ_imp[p,p].  Sign convention: δu = γ_MF - γ_imp.
    """
    u_blocks  = [np.zeros((2, 2))]
    fragments = [[0, 1]]
    gamma_mf  = np.diag([0.5, 0.8, 0.8, 0.5])
    gamma_imp = np.array([[1.3, 0.0], [0.0, 1.5]])  # both > γ_MF → u should decrease
    new_u = update_u_blocks(u_blocks, gamma_mf, [gamma_imp], fragments, step=1.0)
    assert new_u[0][0, 0] < 0.0, "u[0,0] should decrease when γ_imp > γ_MF"
    assert new_u[0][1, 1] < 0.0, "u[1,1] should decrease when γ_imp > γ_MF"


def test_update_u_blocks_result_is_symmetric():
    """u_I must be symmetric even when γ_imp has slight numerical asymmetry."""
    u_blocks  = [np.zeros((2, 2))]
    gamma_mf  = np.diag([1.5, 1.0, 0.8, 1.2])
    # Slightly asymmetric gamma_imp (numerical noise scenario)
    gamma_imp = np.array([[1.0, 0.15], [0.25, 0.5]])
    new_u = update_u_blocks(u_blocks, gamma_mf, [gamma_imp], [[0, 1]], step=1.0)
    np.testing.assert_allclose(new_u[0], new_u[0].T, atol=1e-12,
                                err_msg="updated u block must be symmetric")


def test_update_u_blocks_step_scales_update():
    """Step size must scale the update linearly."""
    u_blocks  = [np.zeros((2, 2))]
    fragments = [[0, 1]]
    gamma_mf  = np.diag([1.5, 1.5, 1.0, 1.0])
    gamma_imp = np.array([[0.5, 0.0], [0.0, 1.0]])  # mf > imp → u > 0
    new_u_s1 = update_u_blocks(u_blocks, gamma_mf, [gamma_imp], fragments, step=1.0)
    new_u_s2 = update_u_blocks(u_blocks, gamma_mf, [gamma_imp], fragments, step=2.0)
    np.testing.assert_allclose(new_u_s2[0], 2.0 * new_u_s1[0], atol=1e-12)


def test_update_u_blocks_zero_delta_unchanged():
    """When γ_imp == γ_MF[frag], u must not change."""
    gamma_mf  = np.diag([1.0, 1.5, 0.8, 0.5])
    gamma_imp = np.array([[1.0, 0.0], [0.0, 1.5]])   # exact match of mf block
    u_init    = np.array([[0.3, 0.1], [0.1, -0.2]])
    new_u = update_u_blocks([u_init], gamma_mf, [gamma_imp], [[0, 1]], step=1.0)
    np.testing.assert_allclose(new_u[0], u_init, atol=1e-12)


# ── damp_u_blocks ────────────────────────────────────────────────────────────

def test_damp_u_blocks_alpha_one_returns_new():
    """alpha=1 → pure new (no damping)."""
    u_new = [np.ones((2, 2))]
    u_old = [np.zeros((2, 2))]
    result = damp_u_blocks(u_new, u_old, alpha=1.0)
    np.testing.assert_allclose(result[0], np.ones((2, 2)))


def test_damp_u_blocks_alpha_zero_returns_old():
    """alpha=0 → pure old (maximum damping / no change)."""
    u_new = [np.ones((2, 2))]
    u_old = [np.full((2, 2), 3.0)]
    result = damp_u_blocks(u_new, u_old, alpha=0.0)
    np.testing.assert_allclose(result[0], np.full((2, 2), 3.0))


def test_damp_u_blocks_half():
    """alpha=0.5 → midpoint."""
    u_new = [np.array([[2.0, 0.0], [0.0, 4.0]])]
    u_old = [np.array([[0.0, 0.0], [0.0, 0.0]])]
    result = damp_u_blocks(u_new, u_old, alpha=0.5)
    np.testing.assert_allclose(result[0], np.array([[1.0, 0.0], [0.0, 2.0]]))


# ── gamma_frag_mismatch ──────────────────────────────────────────────────────

def test_gamma_frag_mismatch_zero_when_equal():
    """Perfect match → mismatch = 0."""
    gamma_mf  = np.diag([1.0, 1.5, 0.8, 1.2])
    gamma_imp = [np.array([[1.0, 0.0], [0.0, 1.5]])]  # exact match of frag [0,1]
    per_frag, global_max = gamma_frag_mismatch(gamma_mf, gamma_imp, [[0, 1]])
    assert global_max < 1e-12
    assert per_frag[0] < 1e-12


def test_gamma_frag_mismatch_correct_value():
    """Mismatch should equal max absolute error on the fragment block."""
    gamma_mf  = np.diag([1.0, 1.5, 0.8, 1.2])
    gamma_imp = [np.array([[1.3, 0.1], [0.1, 1.5]])]  # delta at [0,0]=0.3
    per_frag, global_max = gamma_frag_mismatch(gamma_mf, gamma_imp, [[0, 1]])
    assert global_max == pytest.approx(0.3, abs=1e-12)


def test_gamma_frag_mismatch_multi_fragment():
    """Global max should be the maximum across all fragments."""
    gamma_mf  = np.diag([1.0, 1.5, 0.8, 1.2])
    gamma_imp = [
        np.array([[1.1, 0.0], [0.0, 1.5]]),   # delta = 0.1
        np.array([[1.2, 0.0], [0.0, 1.0]]),   # delta = 0.4 (vs 0.8 and 1.2)
    ]
    per_frag, global_max = gamma_frag_mismatch(gamma_mf, gamma_imp, [[0, 1], [2, 3]])
    assert global_max == pytest.approx(max(per_frag), abs=1e-12)
    assert len(per_frag) == 2


# ── u_blocks_max_change ──────────────────────────────────────────────────────

def test_u_blocks_max_change_zero():
    u = [np.eye(2), np.eye(2)]
    assert u_blocks_max_change(u, u) == pytest.approx(0.0)


def test_u_blocks_max_change_known():
    u_new = [np.array([[2.0, 0.0], [0.0, 3.0]])]
    u_old = [np.array([[1.0, 0.0], [0.0, 1.0]])]
    assert u_blocks_max_change(u_new, u_old) == pytest.approx(2.0)


# ── zero_u_blocks ─────────────────────────────────────────────────────────────

def test_zero_u_blocks_shapes():
    fragments = [[0, 1, 2], [3, 4], [5]]
    u_blocks  = zero_u_blocks(fragments)
    assert len(u_blocks) == 3
    assert u_blocks[0].shape == (3, 3)
    assert u_blocks[1].shape == (2, 2)
    assert u_blocks[2].shape == (1, 1)


def test_zero_u_blocks_all_zero():
    u_blocks = zero_u_blocks([[0, 1, 2], [3, 4, 5]])
    for u in u_blocks:
        np.testing.assert_allclose(u, np.zeros_like(u))
