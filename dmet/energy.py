# TrimCI_Flow/dmet/energy.py
"""
energy.py
=========
DMET fragment energy formulas for 1-shot non-overlapping embedding.

Three formulas are provided:
  A -- 1-body only (no 2-RDM). DEBUG / baseline. Uses h1_phys_proj.
  B -- 1-RDM + 2-RDM fragment partition. PRIMARY. Uses h1_solver.
  C -- Democratic impurity energy. DIAGNOSTIC. Uses h1_solver.

The 2-RDM convention is gated by an RDM energy reconstruction check, which
must pass before any B or C contraction is trusted.
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

    Uses h1_phys_proj (NO Fock/core correction). The A-B difference reveals
    both the missing 2e terms AND the frozen-core Fock correction.

    E_A = sum_{p in frag, q in imp} h1_phys_proj[p,q] * gamma[p,q]
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

    E_I = sum_{p in frag, q in imp} h1_solver[p,q] * gamma[p,q]
        + (1/2) sum_{p in frag, q,r,s in imp} eri_proj[p,q,r,s] * Gamma2[p,q,r,s]

    h1_solver = h1_phys_proj + v_fock_core_imp. The Fock correction captures
    the mean-field interaction between fragment electrons and the frozen core.

    eri convention: chemist notation (pq|rs).
    gamma2 convention: verified via check_2rdm_convention before calling this.
    """
    E_1b = float(np.einsum('pq,pq->', h1_solver[:nfrag, :], gamma[:nfrag, :]))
    E_2b = 0.5 * float(np.einsum('prqs,pqrs->', eri_proj[:nfrag, :, :, :], gamma2[:nfrag, :, :, :]))
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
    No guaranteed ordering relative to A or B.
    """
    n_imp = h1_solver.shape[0]
    E_1b  = float(np.einsum('pq,pq->', h1_solver, gamma))
    E_2b  = 0.5 * float(np.einsum('prqs,pqrs->', eri_proj, gamma2))
    return (E_1b + E_2b) * (nfrag / n_imp)


def rdm_energy_check(
    dets,
    coeffs,
    n_imp: int,
    h1_solver: np.ndarray,
    eri_proj: np.ndarray,
    e_trimci: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Rebuild impurity RDMs and compare their reconstructed energy to TrimCI.

    energy_from_rdm expects physicist ERIs, while eri_proj is stored in
    chemist notation. The transpose below is therefore part of the convention
    check rather than an algorithmic energy correction.
    """
    from trimci.trimci_core import compute_1rdm, compute_2rdm, energy_from_rdm

    gamma  = np.asarray(compute_1rdm(dets, list(coeffs), n_imp)).reshape(n_imp, n_imp)
    gamma2 = np.asarray(compute_2rdm(dets, list(coeffs), n_imp)).reshape(
                 n_imp, n_imp, n_imp, n_imp)

    eri_phys = np.ascontiguousarray(eri_proj.transpose(0, 2, 1, 3))
    e_check = energy_from_rdm(
        gamma.ravel().tolist(),
        gamma2.ravel().tolist(),
        h1_solver.tolist(),        # 2D list -- NOT .ravel()
        eri_phys.ravel().tolist(),
        0.0,
        n_imp,
    )
    discrepancy = abs(float(e_check) - e_trimci)
    return gamma, gamma2, float(e_check), float(discrepancy)


def check_2rdm_convention(
    dets,
    coeffs,
    n_imp: int,
    h1_solver: np.ndarray,
    eri_proj: np.ndarray,
    e_trimci: float,
    tol: float = 1e-6,
    fragment_idx: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Verify that energy_from_rdm(gamma, gamma2, h1_solver, eri_proj, 0.0, n_imp)
    reproduces TrimCI's reported energy within `tol`.

    Raises RuntimeError if the discrepancy exceeds tol. This gates ALL Approach B
    contractions -- if the check fails, no B or C energies are reported.

    Parameters
    ----------
    dets      : TrimCI Determinant C++ objects (from FragmentResult.dets)
    coeffs    : CI coefficients (from FragmentResult.coeffs)
    n_imp     : impurity orbital count (fragment size + bath rank)
    h1_solver : (n_imp, n_imp) impurity h1 including Fock correction
    eri_proj  : (n_imp, n_imp, n_imp, n_imp) projected ERI
    e_trimci  : TrimCI variational energy for this fragment (from result.energy)
    tol       : tolerance in Ha (default 1e-6)

    Returns
    -------
    (gamma, gamma2) : (n_imp, n_imp) and (n_imp, n_imp, n_imp, n_imp) arrays,
                      ready for energy formulas B and C.
    """
    gamma, gamma2, e_check, discrepancy = rdm_energy_check(
        dets, coeffs, n_imp, h1_solver, eri_proj, e_trimci)
    label = f"fragment {fragment_idx}" if fragment_idx is not None else "fragment"
    if discrepancy > tol:
        raise RuntimeError(
            f"2-RDM convention check FAILED on {label}: "
            f"energy_from_rdm={e_check:.8f} Ha, "
            f"TrimCI reported={e_trimci:.8f} Ha, "
            f"discrepancy={discrepancy:.2e} Ha (tol={tol:.0e}). "
            f"Check h1_solver (must include Fock correction) and eri_proj convention."
        )
    print(f"  [DMET] 2-RDM convention check PASSED on {label} (discrepancy={discrepancy:.2e} Ha)")
    return gamma, gamma2
