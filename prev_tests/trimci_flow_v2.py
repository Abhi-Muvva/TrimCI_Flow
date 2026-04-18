"""
trimci_flow_v2.py
=================
Phase B improvements over trimci_flow.py:
  1. Reference-determinant initialization for gamma_mixed (eliminates Iter-1 shock)
  2. Electron-number renormalization of assembled global gamma
  3. J - 0.5K formula (already correct in trimci_flow.py -- preserved here)

Path C (run_fragmented_trimci) is imported from trimci_flow unchanged.
"""
from __future__ import annotations

import numpy as np
from typing import Optional

# Import unchanged components from v1
from TrimCI_Flow.trimci_flow import (
    FragmentedRunResult,
    run_fragmented_trimci,
    compute_fragment_rdm1,
    dress_integrals_meanfield,
)


def _assemble_global_rdm1_diag_v2(fragment_rdm1s, fragment_orbs_list,
                                    n_orb, n_elec, ref_alpha_bits, ref_beta_bits):
    """
    Overlap-aware average of fragment 1-RDM diagonals, with ref-det fallback,
    then electron-number renormalization to conserve total electron count.
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

    # Renormalize to total electron count
    gamma_sum = global_diag.sum()
    if gamma_sum > 0:
        global_diag *= n_elec / gamma_sum
    global_diag = np.clip(global_diag, 0.0, 2.0)
    return global_diag


def run_selfconsistent_fragments_v2(
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
    Self-consistent fragment solve with mean-field coupling (Path B), v2.

    Differences from run_selfconsistent_fragments (v1):
      - gamma_mixed initialized from the reference determinant (not None)
      - Global gamma assembled via _assemble_global_rdm1_diag_v2 with
        electron-number renormalization
      - Damping branch simplified (gamma_mixed always initialized)
    """
    import trimci
    import os
    from TrimCI_Flow.fragment import (
        fragment_by_sliding_window,
        extract_fragment_integrals,
        fragment_electron_count,
    )
    from TrimCI_Flow.trimci_adapter import solve_fragment_trimci

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

    valid = []
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

    # Change 1: Ref-det initialization of gamma_mixed (replaces gamma_mixed = None)
    gamma_ref = np.array(
        [((ref_alpha_bits >> r) & 1) + ((ref_beta_bits >> r) & 1)
         for r in range(n_orb)], dtype=np.float64)
    gamma_mixed = gamma_ref.copy()

    prev_energies = None
    prev_gamma_global = None
    converged = False
    delta_E = float('inf')
    delta_rdm = float('inf')
    final_results = None

    n_elec_total = n_alpha + n_beta

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

            fragment_rdm1s = [
                compute_fragment_rdm1(r.dets, r.coeffs, r.n_orb_frag)
                for r in frag_results
            ]
            fragment_orbs_list = [f[0] for f in valid]
            # Change 3: use _assemble_global_rdm1_diag_v2 with n_elec
            gamma_new = _assemble_global_rdm1_diag_v2(
                fragment_rdm1s, fragment_orbs_list, n_orb, n_elec_total,
                ref_alpha_bits, ref_beta_bits)

            # Change 2: Damping -- always apply (gamma_mixed always initialized)
            gamma_mixed = damping * gamma_new + (1.0 - damping) * gamma_mixed

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
        pass

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
