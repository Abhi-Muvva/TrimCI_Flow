"""
trimci_flow_v3.py
=================
Phase B improvements over trimci_flow_v2.py:
  1. Embedded Iter 0: h1 is dressed from the very first iteration using
     gamma_ref.  Bare-h1 solve is optional and run out-of-band only (not
     part of the SCF trace).  Removes the spurious ~158 Ha first-step jump
     and first-density contamination present in v2.
  2. External gamma-averaging (n_gamma_avg parameter): averages the 1-RDM
     over N independent TrimCI solves per fragment per iteration.  Unlike
     TrimCI's internal num_runs (best-of-N by energy), this is a true
     ensemble-mean estimator of gamma; per-orbital variance drops ~sqrt(N).
  3. Rolling 10-iter std-dev of max|Delta gamma|: logged per iteration.
     Low std-dev = monotone under-relaxation.  High std-dev = noise floor.
  4. Default max_iterations=60, damping=0.1 (v2 values made explicit).

Path C (run_fragmented_trimci) unchanged, imported from trimci_flow.
"""
from __future__ import annotations

import numpy as np
from typing import Optional

from TrimCI_Flow.trimci_flow import (
    FragmentedRunResult,
    run_fragmented_trimci,       # noqa: F401 — re-exported for callers
    compute_fragment_rdm1,
    dress_integrals_meanfield,
)
from TrimCI_Flow.trimci_flow_v2 import _assemble_global_rdm1_diag_v2


def run_selfconsistent_fragments_v3(
    fcidump_path: str,
    window_size: int,
    stride: int,
    max_iterations: int = 60,
    convergence: float = 1e-6,
    rdm_convergence: float = 1e-4,
    damping: float = 0.1,
    n_gamma_avg: int = 1,
    trimci_config: Optional[dict] = None,
    log_bare_reference: bool = True,
) -> FragmentedRunResult:
    """
    Self-consistent fragment solve with mean-field coupling (Path B), v3.

    Parameters
    ----------
    fcidump_path    : path to FCIDUMP file
    window_size     : orbitals per fragment
    stride          : sliding window stride
    max_iterations  : maximum SCF cycles (default 60; v2 used 40)
    convergence     : energy convergence threshold per fragment (Ha)
    rdm_convergence : max|Delta gamma| convergence threshold
    damping         : linear mixing coefficient alpha (0 < alpha <= 1)
    n_gamma_avg     : number of independent TrimCI solves to average for
                      gamma per fragment per iteration.
                      1 = single solve (v2 behaviour).
                      N > 1 = average N RDMs; reduces noise ~sqrt(N).
    trimci_config   : optional TrimCI config overrides
    log_bare_reference : if True, run one bare-h1 solve out-of-band before
                         the SCF loop and log it (Path C comparison).

    Returns
    -------
    FragmentedRunResult with Phase B iteration telemetry.
    result._bare_reference is set to {'energies': [...], 'n_dets': [...]}
    when log_bare_reference=True.

    Differences from v2
    -------------------
    - No `if it == 0: h1_use = h1_bare` bypass.  All SCF iterations use
      dressed h1.  Iter 0 dresses with gamma_ref (pure ref-det field).
    - n_gamma_avg support for external ensemble RDM averaging.
    - rdm_rolling_std tracked and stored in iteration_history.
    """
    import trimci
    import os
    from TrimCI_Flow.fragment import (
        fragment_by_sliding_window,
        extract_fragment_integrals,
        fragment_electron_count,
    )
    from TrimCI_Flow.trimci_adapter import solve_fragment_trimci

    # --- Setup (duplicated from run_fragmented_trimci per D11) ---
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

    n_elec_total = n_alpha + n_beta

    # --- Optional out-of-band bare reference solve ---
    bare_reference = None
    if log_bare_reference:
        bare_energies, bare_dets = [], []
        for (frag_orbs, na, nb, h1_bare, eri_f) in valid:
            res = solve_fragment_trimci(h1_bare, eri_f, na, nb, len(frag_orbs), trimci_config)
            bare_energies.append(res.energy)
            bare_dets.append(res.n_dets)
        bare_reference = {'energies': bare_energies, 'n_dets': bare_dets}
        print(f"  [bare-ref] energies={[f'{e:.4f}' for e in bare_energies]}, "
              f"n_dets={bare_dets}, total={sum(bare_dets)}")

    # --- Initialize gamma_mixed from reference determinant (v2 + v3) ---
    gamma_ref_vec = np.array(
        [((ref_alpha_bits >> r) & 1) + ((ref_beta_bits >> r) & 1)
         for r in range(n_orb)], dtype=np.float64)
    gamma_mixed = gamma_ref_vec.copy()

    # --- Iteration loop ---
    iteration_history = []
    prev_energies     = None
    prev_gamma_global = None
    converged         = False
    delta_E           = float('inf')
    delta_rdm         = float('inf')
    final_results     = None
    delta_rdm_window  = []   # rolling buffer for std-dev

    try:
        for it in range(max_iterations + 1):
            frag_results            = []
            fragment_rdm1s_for_gamma = []

            for (frag_orbs, na, nb, h1_bare, eri_f) in valid:
                # Always dress h1 — embedded Iter 0 (no bare bypass)
                ext_orbs  = [r for r in range(n_orb) if r not in set(frag_orbs)]
                ext_gamma = gamma_mixed[np.asarray(ext_orbs, dtype=np.intp)]
                h1_use    = dress_integrals_meanfield(
                    h1_bare, eri, frag_orbs, ext_gamma, ext_orbs)

                if n_gamma_avg <= 1:
                    res = solve_fragment_trimci(
                        h1_use, eri_f, na, nb, len(frag_orbs), trimci_config)
                    frag_results.append(res)
                    fragment_rdm1s_for_gamma.append(
                        compute_fragment_rdm1(res.dets, res.coeffs, res.n_orb_frag))
                else:
                    # External gamma averaging: N independent solves, average RDMs
                    gamma_sum = None
                    best_res  = None
                    for _ in range(n_gamma_avg):
                        r = solve_fragment_trimci(
                            h1_use, eri_f, na, nb, len(frag_orbs), trimci_config)
                        g = compute_fragment_rdm1(r.dets, r.coeffs, r.n_orb_frag)
                        if gamma_sum is None:
                            gamma_sum = g.copy()
                            best_res  = r
                        else:
                            gamma_sum += g
                            if r.energy < best_res.energy:
                                best_res = r
                    # Use best-energy run for energy/n_dets reporting;
                    # averaged gamma for the SCF update.
                    frag_results.append(best_res)
                    fragment_rdm1s_for_gamma.append(gamma_sum / n_gamma_avg)

            fragment_orbs_list = [f[0] for f in valid]
            gamma_new = _assemble_global_rdm1_diag_v2(
                fragment_rdm1s_for_gamma, fragment_orbs_list, n_orb, n_elec_total,
                ref_alpha_bits, ref_beta_bits)

            # Damping (always applied — gamma_mixed always initialized)
            gamma_mixed = damping * gamma_new + (1.0 - damping) * gamma_mixed

            # Convergence deltas
            energies = [r.energy for r in frag_results]
            n_dets   = [r.n_dets  for r in frag_results]
            if prev_energies is not None:
                delta_E = max(abs(a - b) for a, b in zip(energies, prev_energies))
            if prev_gamma_global is not None:
                delta_rdm = float(np.max(np.abs(gamma_mixed - prev_gamma_global)))

            # Rolling 10-iter std-dev of max|Delta gamma|
            if delta_rdm != float('inf'):
                delta_rdm_window.append(delta_rdm)
                if len(delta_rdm_window) > 10:
                    delta_rdm_window.pop(0)
            rdm_rolling_std = (float(np.std(delta_rdm_window))
                               if len(delta_rdm_window) >= 3 else float('nan'))

            iteration_history.append({
                "iteration":       it,
                "energies":        energies,
                "n_dets":          n_dets,
                "delta_E":         delta_E,
                "delta_rdm":       delta_rdm,
                "rdm_rolling_std": rdm_rolling_std,
            })
            print(f"  Iter {it}: max|dE|={delta_E:.2e}, max|dgamma|={delta_rdm:.2e}, "
                  f"rdm_std10={rdm_rolling_std:.2e}, ndets={sum(n_dets)}")

            prev_energies     = energies
            prev_gamma_global = gamma_mixed.copy()
            final_results     = frag_results

            if it > 0 and delta_E < convergence and delta_rdm < rdm_convergence:
                converged = True
                break
    finally:
        pass

    result = FragmentedRunResult(
        fragment_energies     = [r.energy for r in (final_results or [])],
        fragment_n_dets       = [r.n_dets  for r in (final_results or [])],
        fragment_orbs         = [f[0]      for f in valid],
        total_dets            = sum(r.n_dets for r in (final_results or [])),
        iterations            = len(iteration_history),
        iteration_history     = iteration_history,
        converged             = converged,
        convergence_delta     = delta_E,
        convergence_delta_rdm = delta_rdm,
    )
    result._bare_reference = bare_reference
    return result
