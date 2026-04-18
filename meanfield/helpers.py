"""
meanfield/helpers.py
====================
Core helpers for the meanfield self-consistent loop.

Functions
---------
compute_fragment_rdm1         : 1-RDM from TrimCI wavefunction via C++ binding
dress_integrals_meanfield     : add Fock mean-field correction to fragment h1
assemble_global_rdm1_diag     : overlap-aware gamma assembly with electron-number
                                renormalization
"""
from __future__ import annotations

import numpy as np


def compute_fragment_rdm1(dets: list, coeffs: list, n_orb_frag: int) -> np.ndarray:
    """
    Spin-summed 1-RDM from a TrimCI wavefunction.

    γ[p,q] = Σ_{I,J} c_I c_J <I| a†_p a_q |J>

    Parameters
    ----------
    dets       : TrimCI C++ Determinant objects (from FragmentResult.dets)
    coeffs     : CI coefficients parallel to dets
    n_orb_frag : number of fragment orbitals

    Returns
    -------
    rdm1 : (n_orb_frag, n_orb_frag) float64, spin-summed (alpha + beta)
           diagonal γ[p,p] ∈ [0, 2] gives spatial orbital occupation number
    """
    from trimci.trimci_core import compute_1rdm
    gamma_flat = compute_1rdm(dets, list(coeffs), n_orb_frag)
    return np.asarray(gamma_flat, dtype=np.float64).reshape(n_orb_frag, n_orb_frag)


def dress_integrals_meanfield(
    h1_frag: np.ndarray,
    eri_full: np.ndarray,
    fragment_orbs: list[int],
    external_rdm1_diag: np.ndarray,
    external_orbs: list[int],
) -> np.ndarray:
    """
    Add Fock mean-field correction from external orbitals to h1_frag.

    h1_eff[p,q] = h1_frag[p,q]
                + Σ_{r ∈ external} γ_r · (eri[p,q,r,r] − 0.5·eri[p,r,r,q])

    This is the J − ½K form with spin-summed γ_r ∈ [0,2].
    Note: NOT 2J − K, which would double the shift (that formula is for per-spin γ).

    Parameters
    ----------
    h1_frag            : (n_frag, n_frag) bare fragment one-body integrals
    eri_full           : (n_orb, n_orb, n_orb, n_orb) full-system ERIs, chemist notation
    fragment_orbs      : full-system orbital indices of this fragment
    external_rdm1_diag : (n_external,) spin-summed occupation numbers for external orbitals
    external_orbs      : full-system indices of external orbitals (parallel to above)

    Returns
    -------
    h1_eff : (n_frag, n_frag) dressed one-body integrals, symmetric
    """
    fa = np.asarray(fragment_orbs, dtype=np.intp)
    ea = np.asarray(external_orbs, dtype=np.intp)
    gamma_r = np.asarray(external_rdm1_diag, dtype=np.float64)
    assert gamma_r.shape == (ea.shape[0],), \
        f"gamma length {gamma_r.shape} != external_orbs length {ea.shape}"

    J_block = eri_full[np.ix_(fa, fa, ea, ea)]
    J_diag  = np.diagonal(J_block, axis1=2, axis2=3)
    J_term  = np.einsum('pqr,r->pq', J_diag, gamma_r)

    K_block = eri_full[np.ix_(fa, ea, ea, fa)]
    K_diag  = np.diagonal(K_block, axis1=1, axis2=2)
    K_term  = 0.5 * np.einsum('pqr,r->pq', K_diag, gamma_r)

    h1_eff = h1_frag + J_term - K_term
    assert np.allclose(h1_eff, h1_eff.T, atol=1e-12), \
        "h1_eff lost symmetry — indexing bug in dress_integrals_meanfield"
    return h1_eff


def assemble_global_rdm1_diag(
    fragment_rdm1s: list[np.ndarray],
    fragment_orbs_list: list[list[int]],
    n_orb: int,
    n_elec: int,
    ref_alpha_bits: int,
    ref_beta_bits: int,
) -> np.ndarray:
    """
    Build a global γ diagonal by averaging fragment RDMs over overlapping orbitals,
    then renormalizing to conserve the total electron count.

    Overlap orbitals (shared by two fragments) are averaged, not double-counted.
    Orbitals not covered by any fragment fall back to the reference determinant.

    Parameters
    ----------
    fragment_rdm1s      : list of (n_frag_i, n_frag_i) RDM arrays, one per fragment
    fragment_orbs_list  : list of orbital index lists, parallel to fragment_rdm1s
    n_orb               : total number of orbitals in the full system
    n_elec              : total number of electrons (for renormalization)
    ref_alpha_bits      : uint64 alpha bitstring of reference determinant (fallback)
    ref_beta_bits       : uint64 beta bitstring of reference determinant (fallback)

    Returns
    -------
    global_diag : (n_orb,) float64 spin-summed orbital occupation numbers in [0, 2]
    """
    total = np.zeros(n_orb, dtype=np.float64)
    count = np.zeros(n_orb, dtype=np.int64)
    for gamma_F, frag_orbs in zip(fragment_rdm1s, fragment_orbs_list):
        for local_idx, full_idx in enumerate(frag_orbs):
            total[full_idx] += float(gamma_F[local_idx, local_idx])
            count[full_idx] += 1

    global_diag = np.zeros(n_orb, dtype=np.float64)
    for r in range(n_orb):
        if count[r] > 0:
            global_diag[r] = total[r] / count[r]
        else:
            global_diag[r] = int((ref_alpha_bits >> r) & 1) + int((ref_beta_bits >> r) & 1)

    # Renormalize to conserve total electron count
    gamma_sum = global_diag.sum()
    if gamma_sum > 0:
        global_diag *= n_elec / gamma_sum
    global_diag = np.clip(global_diag, 0.0, 2.0)
    return global_diag
