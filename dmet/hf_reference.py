# TrimCI_Flow/dmet/hf_reference.py
"""
hf_reference.py
===============
Run PySCF UHF on the active-space Hamiltonian from a FCIDUMP to obtain
the mean-field spin-summed 1-RDM (gamma_mf) used for DMET bath construction.

UHF with spin-symmetry breaking is used instead of RHF because Fe4S4 (and
similar strongly correlated open-shell systems) converge RHF to an
all-integer-occupancy closed-shell state that gives incomplete DMET baths
(n_bath < n_frag).  A HOMO-LUMO spin rotation breaks alpha/beta symmetry and
yields the fractional occupancies needed for a full-rank bath.
"""
from __future__ import annotations

import numpy as np


def run_hf(
    h1: np.ndarray,
    eri: np.ndarray,
    n_elec: int,
    n_orb: int,
    n_homo_rotate: int = 3,
    theta: float = 0.3,
) -> tuple[np.ndarray, float]:
    """
    Run PySCF UHF with spin-symmetry breaking on the active-space Hamiltonian.

    A short RHF bootstrap is used to obtain MO coefficients, then HOMO-LUMO
    rotations of ±theta are applied to alpha and beta separately to break
    spin symmetry before the UHF solve.

    Parameters
    ----------
    h1            : (n_orb, n_orb) one-body integrals (chemist notation)
    eri           : (n_orb, n_orb, n_orb, n_orb) two-electron integrals
    n_elec        : total electrons (must be even)
    n_orb         : number of spatial orbitals
    n_homo_rotate : number of HOMO-LUMO pairs to rotate for symmetry breaking
    theta         : rotation angle in radians (default 0.3)

    Returns
    -------
    gamma_mf : (n_orb, n_orb) spin-summed UHF 1-RDM (gamma_alpha + gamma_beta)
    e_hf     : float, UHF total energy

    Warns (print, not raise) if UHF does not converge.
    """
    from pyscf import gto, scf, ao2mo

    n_occ = n_elec // 2
    eri8   = ao2mo.restore(8, eri, n_orb)

    def _make_mol():
        mol = gto.M()
        mol.nelectron = n_elec
        mol.spin      = 0
        mol.verbose   = 0
        return mol

    # ── short RHF bootstrap for MO coefficients ──────────────────────────────
    mol = _make_mol()
    mf_rhf = scf.RHF(mol)
    mf_rhf.get_hcore = lambda *_: h1
    mf_rhf.get_ovlp  = lambda *_: np.eye(n_orb)
    mf_rhf._eri      = eri8
    mf_rhf.max_cycle = 10
    mf_rhf.kernel()
    mo = mf_rhf.mo_coeff   # (n_orb, n_orb)

    # ── spin-broken initial density matrices ─────────────────────────────────
    # Rotate HOMO-LUMO pairs by +theta (alpha) and -theta (beta)
    def _rotate_mo(mo, n_occ, theta_sign):
        mo_rot = mo.copy()
        n_pairs = min(n_homo_rotate, n_occ, n_orb - n_occ)
        for i in range(n_pairs):
            h_idx = n_occ - 1 - i
            l_idx = n_occ + i
            t = theta * theta_sign
            c, s = np.cos(t), np.sin(t)
            col_h = mo_rot[:, h_idx].copy()
            col_l = mo_rot[:, l_idx].copy()
            mo_rot[:, h_idx] =  c * col_h + s * col_l
            mo_rot[:, l_idx] = -s * col_h + c * col_l
        return mo_rot

    mo_a = _rotate_mo(mo, n_occ, +1)
    mo_b = _rotate_mo(mo, n_occ, -1)
    dm_a = mo_a[:, :n_occ] @ mo_a[:, :n_occ].T
    dm_b = mo_b[:, :n_occ] @ mo_b[:, :n_occ].T

    # ── UHF solve ─────────────────────────────────────────────────────────────
    mol = _make_mol()
    mf = scf.UHF(mol)
    mf.get_hcore = lambda *_: h1
    mf.get_ovlp  = lambda *_: np.eye(n_orb)
    mf._eri      = eri8
    mf.max_cycle = 500
    mf.conv_tol  = 1e-8
    mf.kernel([dm_a, dm_b])

    if not mf.converged:
        print("[DMET WARNING] UHF did not converge — gamma_mf may be unreliable")

    dm_uhf   = mf.make_rdm1()          # shape (2, n_orb, n_orb)
    gamma_mf = dm_uhf[0] + dm_uhf[1]  # spin-summed, shape (n_orb, n_orb)
    return gamma_mf, float(mf.e_tot)
