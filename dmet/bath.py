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
    n_elec_core    : float, Tr(gamma_core) -- electrons in the frozen core.
    n_bath         : int, number of significant bath orbitals.
    env_orbs       : list[int], environment orbital indices (full-system space).
    """
    frag_set = set(frag_orbs)
    env_orbs = [r for r in range(n_orb) if r not in frag_set]
    n_frag   = len(frag_orbs)
    n_env    = len(env_orbs)

    # frag-env block of gamma_mf: (n_frag, n_env)
    gamma_AB = gamma_mf[np.ix_(frag_orbs, env_orbs)]

    # Full SVD -- Vt is (n_env, n_env)
    # Rows 0:n_bath of Vt -> bath subspace
    # Rows n_bath:   of Vt -> orthogonal complement (core)
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

    # Reconstruct in full orbital space -- non-zero only in env_orbs block
    gamma_core_env_blk = core_vecs_env @ gamma_core_local @ core_vecs_env.T  # (n_env, n_env)
    gamma_core_full = np.zeros((n_orb, n_orb))
    gamma_core_full[np.ix_(env_orbs, env_orbs)] = gamma_core_env_blk

    n_elec_core = float(np.trace(gamma_core_local))

    return P_imp, gamma_core_full, n_elec_core, n_bath, env_orbs


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
                   Used in energy formula A.
    h1_solver    : (n_imp, n_imp) h1_phys_proj + v_fock_core_imp.
                   Used as input to TrimCI AND in energy formulas B and C.
    eri_proj     : (n_imp, n_imp, n_imp, n_imp) physical ERIs in impurity space.
    """
    n_orb = h1.shape[0]

    # Physical h1 projected to impurity
    h1_phys_proj = P_imp.T @ h1 @ P_imp   # (n_imp, n_imp)

    # ERI projection: four sequential contractions
    T = np.einsum('pqrs,pi->iqrs', eri, P_imp)
    T = np.einsum('iqrs,qj->ijrs', T,   P_imp)
    T = np.einsum('ijrs,rk->ijks', T,   P_imp)
    eri_proj = np.einsum('ijks,sl->ijkl', T, P_imp)   # (n_imp, n_imp, n_imp, n_imp)

    # Fock embedding from frozen core (explicit np.ix_ blocks)
    env = np.array(env_orbs)
    gamma_env_blk = gamma_core_full[np.ix_(env, env)]                          # (n_env, n_env)
    arange        = np.arange(n_orb)
    eri_J = eri[np.ix_(arange, arange, env, env)]                              # (n_orb, n_orb, n_env, n_env)
    eri_K = eri[np.ix_(arange, env,    env, arange)]                           # (n_orb, n_env, n_env, n_orb)
    J_full = np.einsum('rs,pqrs->pq', gamma_env_blk, eri_J)                    # (n_orb, n_orb)
    K_full = np.einsum('rs,prsq->pq', gamma_env_blk, eri_K)                    # (n_orb, n_orb)
    v_fock_full = J_full - 0.5 * K_full
    v_fock_imp  = P_imp.T @ v_fock_full @ P_imp                                # (n_imp, n_imp)

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
    n_elec_core  : Tr(gamma_core) -- electrons in the frozen core (float)

    Returns
    -------
    (n_alpha_imp, n_beta_imp) : both equal to n_elec_imp // 2
    """
    # Round to nearest even integer (standard in DMET with fractional-occupancy
    # references such as UHF, which give non-integer n_elec_core).
    n_elec_imp_float = n_elec_total - n_elec_core
    n_elec_imp = 2 * int(round(n_elec_imp_float / 2))
    if abs(n_elec_imp_float - n_elec_imp) > 0.5:
        print(f"  [DMET WARNING] n_elec_imp rounding: {n_elec_imp_float:.4f} -> {n_elec_imp}")
    n_half = n_elec_imp // 2
    return n_half, n_half
