# TrimCI_Flow/dmet/correlation_potential.py
"""
correlation_potential.py
========================
Utilities for the SC-DMET correlation potential u.

u is a block-diagonal operator in the fragment space:
  u_emb[frag_I, frag_I] = u_I   (symmetric n_frag × n_frag matrix)
  u_emb = 0 elsewhere

It is added to h1 before the UHF reference calculation to enforce 1-RDM
matching between the impurity solver and the reference on each fragment block:

  γ_UHF(h1 + u_emb)[frag_I, frag_I]  →  γ_imp_I[0:n_frag, 0:n_frag]

Convergence is declared when this mismatch is below conv_tol.

All functions here are pure numpy — no TrimCI or PySCF imports.
"""
from __future__ import annotations

import numpy as np


def build_u_emb(
    u_blocks: list[np.ndarray],
    fragments: list[list[int]],
    n_orb: int,
) -> np.ndarray:
    """
    Assemble the full-space correlation potential from per-fragment blocks.

    Parameters
    ----------
    u_blocks  : list of (n_frag_I, n_frag_I) symmetric matrices, one per fragment
    fragments : list of orbital index lists (same ordering as u_blocks)
    n_orb     : total number of orbitals

    Returns
    -------
    u_emb : (n_orb, n_orb) symmetric matrix; zero outside fragment diagonal blocks
    """
    u_emb = np.zeros((n_orb, n_orb))
    for u_I, frag_orbs in zip(u_blocks, fragments):
        idx = np.array(frag_orbs)
        u_emb[np.ix_(idx, idx)] += u_I
    return u_emb


def extract_gamma_frag(gamma: np.ndarray, frag_orbs: list[int]) -> np.ndarray:
    """
    Extract the fragment-diagonal block of a density matrix.

    Parameters
    ----------
    gamma     : (n, n) density matrix (full system or impurity space)
    frag_orbs : orbital indices to extract (in the gamma index basis)

    Returns
    -------
    (n_frag, n_frag) subblock of gamma at frag_orbs × frag_orbs
    """
    idx = np.array(frag_orbs)
    return gamma[np.ix_(idx, idx)]


def update_u_blocks(
    u_blocks: list[np.ndarray],
    gamma_mf: np.ndarray,
    gamma_imp_frags: list[np.ndarray],
    fragments: list[list[int]],
    step: float = 1.0,
) -> list[np.ndarray]:
    """
    Direct-fitting gradient step: u_I += step * (γ_MF[frag_I] - γ_imp_I).

    This is the standard SC-DMET direct fitting step. It drives γ_MF toward
    the correlated γ_imp target on each fragment block. At SC convergence
    γ_imp = γ_MF[frag], so Δu → 0.

    The update is symmetrized to prevent drift from numerical noise in γ_imp.

    Parameters
    ----------
    u_blocks        : current per-fragment u matrices (list of (n_frag_I, n_frag_I))
    gamma_mf        : (n_orb, n_orb) current UHF reference density matrix
    gamma_imp_frags : list of (n_frag_I, n_frag_I) impurity γ on the fragment block
                      (indices 0:n_frag_I in impurity space = fragment orbitals)
    fragments       : list of fragment orbital index lists (full-system indices)
    step            : gradient descent step size (default 1.0)

    Returns
    -------
    new_u_blocks : updated list of (n_frag_I, n_frag_I) symmetric matrices
    """
    new_u_blocks = []
    for u_I, gamma_imp_I, frag_orbs in zip(u_blocks, gamma_imp_frags, fragments):
        gamma_mf_I = extract_gamma_frag(gamma_mf, frag_orbs)
        # Sign convention: raising u[p,p] raises orbital energy ε_p → fewer electrons
        # → γ_MF[p,p] decreases. To drive γ_MF → γ_imp:
        #   γ_MF > γ_imp (UHF has too many electrons): δu > 0  (raise energy, shed electrons)
        #   γ_MF < γ_imp (UHF has too few electrons):  δu < 0  (lower energy, attract electrons)
        # Therefore: δu = step * (γ_MF - γ_imp)   [NOT (γ_imp - γ_MF)]
        delta = gamma_mf_I - gamma_imp_I
        # symmetrize to prevent drift from off-diagonal numerical noise
        u_new_I = u_I + step * 0.5 * (delta + delta.T)
        new_u_blocks.append(u_new_I)
    return new_u_blocks


def damp_u_blocks(
    u_blocks_new: list[np.ndarray],
    u_blocks_old: list[np.ndarray],
    alpha: float,
) -> list[np.ndarray]:
    """
    Linear mixing (damping) for the correlation potential update.

    u_mixed = alpha * u_new + (1 - alpha) * u_old

    Parameters
    ----------
    u_blocks_new : proposed new u_blocks (from update_u_blocks)
    u_blocks_old : previous u_blocks
    alpha        : mixing fraction in (0, 1]. alpha=1.0 = no damping.

    Returns
    -------
    mixed u_blocks list
    """
    return [
        alpha * u_new + (1.0 - alpha) * u_old
        for u_new, u_old in zip(u_blocks_new, u_blocks_old)
    ]


def gamma_frag_mismatch(
    gamma_mf: np.ndarray,
    gamma_imp_frags: list[np.ndarray],
    fragments: list[list[int]],
) -> tuple[list[float], float]:
    """
    Compute max|γ_imp_I - γ_MF[frag_I]| per fragment and globally.

    This is the SC-DMET convergence metric: when it falls below conv_tol,
    the embedding is self-consistent and E_B is physically meaningful.

    Parameters
    ----------
    gamma_mf        : (n_orb, n_orb) UHF reference density matrix
    gamma_imp_frags : list of (n_frag_I, n_frag_I) impurity γ on fragment block
    fragments       : list of fragment orbital index lists

    Returns
    -------
    per_frag_max : list of per-fragment max|Δγ|
    global_max   : max over all fragments
    """
    per_frag = []
    for gamma_imp_I, frag_orbs in zip(gamma_imp_frags, fragments):
        gamma_mf_I = extract_gamma_frag(gamma_mf, frag_orbs)
        per_frag.append(float(np.max(np.abs(gamma_imp_I - gamma_mf_I))))
    return per_frag, max(per_frag)


def u_blocks_max_change(
    u_blocks_new: list[np.ndarray],
    u_blocks_old: list[np.ndarray],
) -> float:
    """
    Compute max|u_new - u_old| across all fragment blocks.

    Used as a secondary convergence diagnostic (correlation potential stability).
    """
    diffs = [float(np.max(np.abs(u_n - u_o)))
             for u_n, u_o in zip(u_blocks_new, u_blocks_old)]
    return max(diffs)


def zero_u_blocks(fragments: list[list[int]]) -> list[np.ndarray]:
    """
    Initialize u_blocks to zero (one zero matrix per fragment).

    Parameters
    ----------
    fragments : list of fragment orbital index lists

    Returns
    -------
    list of zero (n_frag_I, n_frag_I) float64 arrays
    """
    return [np.zeros((len(frag), len(frag))) for frag in fragments]
