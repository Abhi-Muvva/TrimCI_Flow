"""
trimci_flow.py
==============
Main driver: fragmented TrimCI solve with optional inter-fragment coupling.

Three coupling levels (implementation order):
  Path C — no coupling (uncoupled baseline)   [Phase 3]
  Path B — mean-field embedding (DMET-style)  [Phase 4]
  Path A — BCH integral dressing (full QFlow) [Phase 5, future]

All functions use CHEMIST-notation integrals: eri[p,q,r,s] = (pq|rs).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class FragmentedRunResult:
    """Aggregated result from a fragmented TrimCI run."""
    fragment_energies: list[float]
    fragment_n_dets: list[int]
    fragment_orbs: list[list[int]]
    total_dets: int                      # sum of fragment_n_dets
    brute_force_dets: int = 10095        # Fe4S4 reference
    iterations: int = 1                  # >1 for Path B self-consistent
    # Phase B additions (default values preserve Path C behaviour)
    iteration_history: list = field(default_factory=list)
    converged: bool = False
    convergence_delta: float = float('inf')
    convergence_delta_rdm: float = float('inf')


# ---------------------------------------------------------------------------
# Path C: Uncoupled fragmentation baseline
# ---------------------------------------------------------------------------

def run_fragmented_trimci(
    fcidump_path: str,
    window_size: int = 15,
    stride: int = 10,
    trimci_config: Optional[dict] = None,
) -> FragmentedRunResult:
    """
    Fragment-and-solve without inter-fragment coupling (Path C).

    Steps
    -----
    1. Read FCIDUMP (h1, eri in chemist notation, n_elec, n_orb, E_nuc)
    2. Sort orbitals by h1 diagonal (orbital energy)
    3. Create fragments via sliding window
    4. For each fragment: extract integrals, count electrons, run TrimCI
    5. Return aggregated result with per-fragment energies and det counts

    Parameters
    ----------
    fcidump_path  : path to FCIDUMP file (e.g. fcidump_cycle_6)
    window_size   : orbitals per fragment (e.g. 15)
    stride        : sliding window stride (e.g. 10)
    trimci_config : optional TrimCI config overrides

    Returns
    -------
    FragmentedRunResult

    Notes
    -----
    - Summing fragment energies is NOT meaningful (double-counting). Only
      compare total_dets vs brute_force_dets.
    - Fe4S4 reference: 10,095 determinants, E ≈ −327.1920 Ha, E_nuc = 0.
    """
    # Imports kept inside function so the module can be imported even before
    # fragment.py and trimci_adapter.py are fully implemented.
    import trimci
    from TrimCI_Flow.fragment import (
        fragment_by_sliding_window,
        extract_fragment_integrals,
        fragment_electron_count,
    )
    from TrimCI_Flow.trimci_adapter import solve_fragment_trimci

    # Step 1: Read FCIDUMP
    h1, eri, n_elec, n_orb, E_nuc, n_alpha, n_beta, psym = trimci.read_fcidump(fcidump_path)

    # Step 2: Sort orbitals by h1 diagonal (ascending energy)
    order = np.argsort(np.diag(h1))

    # Step 3: Build reference bitstrings for electron counting.
    # Prefer the actual correlated reference determinant from dets.npz (same dir as FCIDUMP),
    # which has physically correct open-shell occupations — avoids trivially-occupied fragments
    # that arise when the HF reference fully fills a low-energy fragment window.
    # Fall back to HF approximation (lowest n_alpha/n_beta by orbital energy) if absent.
    import os
    _dets_path = os.path.join(os.path.dirname(os.path.abspath(fcidump_path)), "dets.npz")
    if os.path.exists(_dets_path):
        _ref_data = np.load(_dets_path)
        ref_alpha_bits = int(_ref_data["dets"][0, 0])  # uint64 alpha bitstring of det 0
        ref_beta_bits  = int(_ref_data["dets"][0, 1])  # uint64 beta  bitstring of det 0
    else:
        ref_alpha_bits = int(sum(1 << int(order[i]) for i in range(n_alpha)))
        ref_beta_bits  = int(sum(1 << int(order[i]) for i in range(n_beta)))

    # Step 4: Create fragments
    fragments = fragment_by_sliding_window(n_orb, order, window_size, stride)

    # Step 5: Solve each fragment
    fragment_energies = []
    fragment_n_dets   = []
    fragment_orbs_out = []

    for frag_orbs in fragments:
        na, nb = fragment_electron_count(ref_alpha_bits, ref_beta_bits, frag_orbs)
        n_frag = len(frag_orbs)

        # Skip degenerate fragments (empty spin channel or more electrons than orbitals)
        if na == 0 or nb == 0 or na > n_frag or nb > n_frag:
            print(f"  Skipping fragment {frag_orbs[:3]}... "
                  f"(n_alpha={na}, n_beta={nb}, n_orb={n_frag})")
            continue

        h1_f, eri_f = extract_fragment_integrals(h1, eri, frag_orbs)
        result = solve_fragment_trimci(h1_f, eri_f, na, nb, n_frag, trimci_config)

        fragment_energies.append(result.energy)
        fragment_n_dets.append(result.n_dets)
        fragment_orbs_out.append(frag_orbs)
        print(f"  Fragment orbs {frag_orbs[0]}..{frag_orbs[-1]}: "
              f"n_dets={result.n_dets}, energy={result.energy:.4f} Ha")

    # NOTE: do NOT sum fragment_energies — double-counting makes it physically
    # meaningless in Path C. Compare total_dets vs brute_force_dets instead.
    return FragmentedRunResult(
        fragment_energies=fragment_energies,
        fragment_n_dets=fragment_n_dets,
        fragment_orbs=fragment_orbs_out,
        total_dets=sum(fragment_n_dets),
    )


# ---------------------------------------------------------------------------
# Path B: Mean-field coupling (self-consistent)
# ---------------------------------------------------------------------------

def compute_fragment_rdm1(
    dets: list,
    coeffs: list,
    n_orb_frag: int,
) -> np.ndarray:
    """
    Compute one-particle reduced density matrix from a TrimCI wavefunction.

    γ[p,q] = Σ_{I,J} c_I c_J <I| a†_p a_q |J>

    Parameters
    ----------
    dets      : list of TrimCI Determinant objects
    coeffs    : CI coefficients (parallel to dets)
    n_orb_frag: number of orbitals in this fragment

    Returns
    -------
    rdm1 : (n_orb_frag, n_orb_frag) array, spin-summed (alpha + beta)

    Notes
    -----
    The diagonal γ[p,p] gives the occupation number of orbital p.
    For a closed-shell reference, γ[p,p] ∈ [0, 2].
    """
    from trimci.trimci_core import compute_1rdm
    gamma_flat = compute_1rdm(dets, list(coeffs), n_orb_frag)
    gamma = np.asarray(gamma_flat, dtype=np.float64).reshape(n_orb_frag, n_orb_frag)
    return gamma


def dress_integrals_meanfield(
    h1_frag: np.ndarray,
    eri_full: np.ndarray,
    fragment_orbs: list[int],
    external_rdm1_diag: np.ndarray,
    external_orbs: list[int],
) -> np.ndarray:
    """
    Add mean-field correction from external orbitals to h1_frag.

    h1_eff[p,q] = h1_frag[p,q]
                  + Σ_{r ∈ external} γ_r * (eri[p,q,r,r] - 0.5*eri[p,r,r,q])

    γ_r is the spin-summed diagonal of the external 1-RDM (ranges 0→2 for
    spatial orbitals).  The coefficient is (J − ½K), NOT (2J − K): the factor
    of 2 in front of J would only be correct if γ_r were a per-spin occupation
    (0→1).  Using spin-summed γ with 2J−K doubles the intended mean-field shift.

    where indices p,q are in the fragment orbital basis and r is in the
    full-system basis (external_orbs).

    Parameters
    ----------
    h1_frag          : (n_frag, n_frag) fragment one-body integrals
    eri_full         : (n_orb, n_orb, n_orb, n_orb) full-system ERIs,
                       chemist notation
    fragment_orbs    : orbital indices of this fragment in the full system
    external_rdm1_diag : 1D array of spin-summed occupation numbers for external
                         orbitals (diagonal of γ_α+γ_β, length n_external)
    external_orbs    : orbital indices of the external orbitals in the full system

    Returns
    -------
    h1_eff : (n_frag, n_frag) dressed one-body integrals
    """
    fa = np.asarray(fragment_orbs, dtype=np.intp)
    ea = np.asarray(external_orbs, dtype=np.intp)
    gamma_r = np.asarray(external_rdm1_diag, dtype=np.float64)
    assert gamma_r.shape == (ea.shape[0],), \
        f"gamma length {gamma_r.shape} != external_orbs length {ea.shape}"
    # J term: sum_r gamma_r * eri[p_full, q_full, r, r]  (spin-summed γ, no factor-2)
    J_block = eri_full[np.ix_(fa, fa, ea, ea)]         # (nF, nF, nE, nE)
    J_diag  = np.diagonal(J_block, axis1=2, axis2=3)   # (nF, nF, nE)
    J_term  = np.einsum('pqr,r->pq', J_diag, gamma_r)
    # K term: 0.5 * sum_r gamma_r * eri[p_full, r, r, q_full]  (spin-summed γ)
    K_block = eri_full[np.ix_(fa, ea, ea, fa)]         # (nF, nE, nE, nF)
    K_diag  = np.diagonal(K_block, axis1=1, axis2=2)   # (nF, nF, nE)
    K_term  = 0.5 * np.einsum('pqr,r->pq', K_diag, gamma_r)
    h1_eff  = h1_frag + J_term - K_term
    assert np.allclose(h1_eff, h1_eff.T, atol=1e-12), \
        "h1_eff lost symmetry -- indexing bug in dress_integrals_meanfield"
    return h1_eff


def _assemble_global_rdm1_diag(fragment_rdm1s, fragment_orbs_list,
                                n_orb, ref_alpha_bits, ref_beta_bits):
    """Overlap-aware average of fragment 1-RDM diagonals, with ref-det fallback."""
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
    return global_diag


def run_selfconsistent_fragments(
    fcidump_path: str,
    window_size: int,
    stride: int,
    max_iterations: int = 20,
    convergence: float = 1e-6,
    rdm_convergence: float = 1e-4,
    damping: float = 0.5,
    trimci_config: Optional[dict] = None,
) -> FragmentedRunResult:
    """
    Self-consistent fragment solve with mean-field coupling (Path B).

    Algorithm
    ---------
    1. Read FCIDUMP, create fragments (same as Path C)
    2. Initial solve: each fragment with bare (undressed) integrals
    3. Compute 1-RDMs for each fragment
    4. For each fragment: dress h1 with mean-field from all other fragments
    5. Re-solve each fragment with dressed integrals
    6. Check convergence: max|ΔE_frag| < convergence AND max|Δγ| < rdm_convergence
    7. Repeat 3-6 until converged or max_iterations reached

    Parameters
    ----------
    fcidump_path    : path to FCIDUMP file
    window_size     : orbitals per fragment
    stride          : sliding window stride
    max_iterations  : maximum self-consistency cycles
    convergence     : energy convergence threshold (Ha) per fragment
    rdm_convergence : RDM diagonal convergence threshold
    damping         : linear mixing coefficient alpha (0 < alpha <= 1.0);
                      alpha=0.5 default; alpha=1.0 disables mixing
    trimci_config   : optional TrimCI config overrides

    Returns
    -------
    FragmentedRunResult with Phase B iteration telemetry
    """
    import trimci
    import os
    from TrimCI_Flow.fragment import (
        fragment_by_sliding_window,
        extract_fragment_integrals,
        fragment_electron_count,
    )
    from TrimCI_Flow.trimci_adapter import solve_fragment_trimci

    # --- Setup (duplicated from run_fragmented_trimci:80-117, NOT refactored per D11) ---
    h1, eri, n_elec, n_orb, E_nuc, n_alpha, n_beta, psym = trimci.read_fcidump(fcidump_path)
    order = np.argsort(np.diag(h1))

    _dets_path = os.path.join(os.path.dirname(os.path.abspath(fcidump_path)), "dets.npz")
    if os.path.exists(_dets_path):
        _ref_data = np.load(_dets_path)
        ref_alpha_bits = int(_ref_data["dets"][0, 0])
        ref_beta_bits  = int(_ref_data["dets"][0, 1])
    else:
        ref_alpha_bits = int(sum(1 << int(order[i]) for i in range(n_alpha)))
        ref_beta_bits  = int(sum(1 << int(order[i]) for i in range(n_beta)))

    fragments_all = fragment_by_sliding_window(n_orb, order, window_size, stride)

    # Filter degenerate fragments; freeze (na, nb) forever (D8).
    valid = []  # list of (frag_orbs, na, nb, h1_frag_bare, eri_frag)
    for frag_orbs in fragments_all:
        na, nb = fragment_electron_count(ref_alpha_bits, ref_beta_bits, frag_orbs)
        n_frag = len(frag_orbs)
        if na == 0 or nb == 0 or na > n_frag or nb > n_frag:
            print(f"  Skipping fragment {frag_orbs[:3]}... (na={na}, nb={nb}, n={n_frag})")
            continue
        h1_f, eri_f = extract_fragment_integrals(h1, eri, frag_orbs)
        valid.append((frag_orbs, na, nb, h1_f, eri_f))

    # --- Iteration loop ---
    iteration_history = []
    gamma_mixed = None
    prev_energies = None
    prev_gamma_global = None
    converged = False
    delta_E = float('inf')
    delta_rdm = float('inf')
    final_results = None

    try:
        for it in range(max_iterations + 1):
            frag_results = []
            for (frag_orbs, na, nb, h1_bare, eri_f) in valid:
                if it == 0:
                    h1_use = h1_bare
                else:
                    ext_orbs = [r for r in range(n_orb) if r not in set(frag_orbs)]
                    ext_gamma = gamma_mixed[np.asarray(ext_orbs, dtype=np.intp)]
                    h1_use = dress_integrals_meanfield(
                        h1_bare, eri, frag_orbs, ext_gamma, ext_orbs)
                res = solve_fragment_trimci(h1_use, eri_f, na, nb, len(frag_orbs), trimci_config)
                frag_results.append(res)

            # Build gamma_new from this iteration's RDMs (D2 + D3).
            fragment_rdm1s = [
                compute_fragment_rdm1(r.dets, r.coeffs, r.n_orb_frag)
                for r in frag_results
            ]
            fragment_orbs_list = [f[0] for f in valid]
            gamma_new = _assemble_global_rdm1_diag(
                fragment_rdm1s, fragment_orbs_list, n_orb,
                ref_alpha_bits, ref_beta_bits)

            # Damping (D6).
            if gamma_mixed is None:
                gamma_mixed = gamma_new.copy()      # iteration 0: no mixing yet
            else:
                gamma_mixed = damping * gamma_new + (1.0 - damping) * gamma_mixed

            # Compute deltas (D4).
            energies = [r.energy for r in frag_results]
            n_dets   = [r.n_dets for r in frag_results]
            if prev_energies is not None:
                delta_E = max(abs(a - b) for a, b in zip(energies, prev_energies))
            if prev_gamma_global is not None:
                delta_rdm = float(np.max(np.abs(gamma_mixed - prev_gamma_global)))

            iteration_history.append({
                "iteration": it,
                "energies":  energies,
                "n_dets":    n_dets,
                "delta_E":   delta_E,
                "delta_rdm": delta_rdm,
            })
            print(f"  Iter {it}: max|dE|={delta_E:.2e}, max|dgamma|={delta_rdm:.2e}, "
                  f"ndets_total={sum(n_dets)}")

            prev_energies     = energies
            prev_gamma_global = gamma_mixed.copy()
            final_results     = frag_results

            if it > 0 and delta_E < convergence and delta_rdm < rdm_convergence:
                converged = True
                break
    finally:
        pass  # partial results remain in final_results / iteration_history

    return FragmentedRunResult(
        fragment_energies     = [r.energy for r in (final_results or [])],
        fragment_n_dets       = [r.n_dets for r in (final_results or [])],
        fragment_orbs         = [f[0] for f in valid],
        total_dets            = sum(r.n_dets for r in (final_results or [])),
        iterations            = len(iteration_history),
        iteration_history     = iteration_history,
        converged             = converged,
        convergence_delta     = delta_E,
        convergence_delta_rdm = delta_rdm,
    )
