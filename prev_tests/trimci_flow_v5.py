"""
trimci_flow_v5.py
=================
Phase B improvement over trimci_flow_v3.py:
  Owner-fragment gamma assembly replaces overlap-averaging.

Scientific motivation (from orbital diagnostic Run 9):
  Orbital 33 (F0∩F1 overlap) drives max|Δγ| in 16/20 SCF iterations and is
  SYSTEMATIC (std/mean = 0.65 < 1), not stochastic.  It sits near the high-energy
  edge of F0's window and the low-energy edge of F1's window.  The two fragments
  solve different embedded Hamiltonians and persistently disagree on gamma[33].
  Overlap-averaging creates an oscillating value that satisfies neither fragment's
  SCF condition.

  Owner-fragment assignment: each orbital belongs to exactly ONE fragment — the one
  whose energy-window center is closest in rank.  This eliminates the systematic
  F0-vs-F1 disagreement without requiring any additional TrimCI solves.

  The stochastic component (excl_F1 orbitals 28, 4, 9) is addressed by keeping
  n_gamma_avg=5 from v3.

All other v3 improvements are retained:
  - Embedded Iter 0 (gamma_ref initialization, no bare bypass)
  - n_gamma_avg for external ensemble RDM averaging
  - Rolling 10-iter std-dev of max|Δγ|
  - Default max_iterations=60, damping=0.1

Path C (run_fragmented_trimci) unchanged, imported from trimci_flow.
"""
from __future__ import annotations

import numpy as np
from typing import Optional

from TrimCI_Flow.trimci_flow import (
    FragmentedRunResult,
    run_fragmented_trimci,       # noqa: F401 — re-exported
    compute_fragment_rdm1,
    dress_integrals_meanfield,
)


def _compute_owner_map(fragment_orbs_list, n_orb, order):
    """
    For each orbital r in 0..n_orb-1, determine which fragment owns it.

    Owner = fragment with the smallest distance from r's energy rank to the
    fragment's center rank (= midpoint of its rank range).
    Uncovered orbitals return -1.

    Parameters
    ----------
    fragment_orbs_list : list of lists of orbital indices, one per fragment
    n_orb  : total number of orbitals
    order  : np.ndarray of shape (n_orb,); order[i] = orbital index of i-th
             lowest diagonal-h1 orbital (output of np.argsort(np.diag(h1)))

    Returns
    -------
    owner : np.ndarray of shape (n_orb,) with int values in {-1, 0, .., n_frag-1}
    """
    # energy rank of each orbital: rank_of[r] = position of orbital r in sorted order
    rank_of = np.empty(n_orb, dtype=int)
    for i, r in enumerate(order):
        rank_of[r] = i

    # precompute center rank for each fragment
    frag_center_rank = []
    for frag_orbs in fragment_orbs_list:
        ranks = [rank_of[r] for r in frag_orbs]
        frag_center_rank.append(0.5 * (min(ranks) + max(ranks)))

    # assign owner
    owner = np.full(n_orb, -1, dtype=int)
    for r in range(n_orb):
        r_rank = rank_of[r]
        best_fi, best_dist = -1, float('inf')
        for fi, frag_orbs in enumerate(fragment_orbs_list):
            if r in set(frag_orbs):
                dist = abs(float(r_rank) - frag_center_rank[fi])
                if dist < best_dist:
                    best_dist = dist
                    best_fi = fi
        owner[r] = best_fi
    return owner


def _assemble_global_rdm1_owner_v5(
    fragment_rdm1s,
    fragment_orbs_list,
    n_orb,
    n_elec,
    ref_alpha_bits,
    ref_beta_bits,
    owner,
    local_idx_map,
):
    """
    Owner-fragment gamma assembly: each orbital gets its gamma from exactly one
    fragment (its owner), with no overlap-averaging.

    Followed by electron-number renormalization (same as v2) and clip to [0,2].

    Parameters
    ----------
    fragment_rdm1s    : list of (n_frag_orb, n_frag_orb) RDM matrices
    fragment_orbs_list: list of fragment orbital index lists
    n_orb             : total orbitals
    n_elec            : total electrons (for renormalization)
    ref_alpha_bits    : reference determinant alpha bits (fallback for uncovered orbs)
    ref_beta_bits     : reference determinant beta bits
    owner             : np.ndarray shape (n_orb,); owner[r] = fragment index or -1
    local_idx_map     : dict (fi, r) -> local_idx in fragment_rdm1s[fi]

    Returns
    -------
    global_diag : np.ndarray shape (n_orb,)
    """
    global_diag = np.zeros(n_orb, dtype=np.float64)
    for r in range(n_orb):
        fi = int(owner[r])
        if fi < 0:
            global_diag[r] = int((ref_alpha_bits >> r) & 1) + int((ref_beta_bits >> r) & 1)
        else:
            local_idx = local_idx_map[(fi, r)]
            global_diag[r] = float(fragment_rdm1s[fi][local_idx, local_idx])

    # Electron-number renormalization (conserves total electron count)
    gamma_sum = global_diag.sum()
    if gamma_sum > 0:
        global_diag *= n_elec / gamma_sum
    global_diag = np.clip(global_diag, 0.0, 2.0)
    return global_diag


def run_selfconsistent_fragments_v5(
    fcidump_path: str,
    window_size: int,
    stride: int,
    max_iterations: int = 60,
    convergence: float = 1e-6,
    rdm_convergence: float = 1e-4,
    damping: float = 0.1,
    n_gamma_avg: int = 5,
    trimci_config: Optional[dict] = None,
    log_bare_reference: bool = True,
) -> FragmentedRunResult:
    """
    Self-consistent fragment solve with owner-fragment gamma assembly (Path B), v5.

    Parameters
    ----------
    fcidump_path    : path to FCIDUMP file
    window_size     : orbitals per fragment
    stride          : sliding window stride
    max_iterations  : maximum SCF cycles (default 60)
    convergence     : energy convergence threshold per fragment (Ha)
    rdm_convergence : max|Delta gamma| convergence threshold
    damping         : linear mixing coefficient alpha (0 < alpha <= 1)
    n_gamma_avg     : independent TrimCI solves averaged per fragment per iter
    trimci_config   : optional TrimCI config overrides
    log_bare_reference : run bare-h1 solve out-of-band and log (Path C ref)

    Differences from v3
    -------------------
    - gamma assembly uses _assemble_global_rdm1_owner_v5 (owner-fragment
      assignment, not overlap-averaging).
    - owner_map precomputed once: orbital r → fragment whose energy-window
      center is closest in rank to r.
    - All other v3 features retained (embedded iter 0, n_gamma_avg,
      rolling std, damping=0.1 default).
    """
    import trimci
    import os
    from TrimCI_Flow.fragment import (
        fragment_by_sliding_window,
        extract_fragment_integrals,
        fragment_electron_count,
    )
    from TrimCI_Flow.trimci_adapter import solve_fragment_trimci

    # ---- Setup ----
    h1, eri, n_elec, n_orb, E_nuc, n_alpha, n_beta, psym = trimci.read_fcidump(fcidump_path)
    order = np.argsort(np.diag(h1))

    _dets_path = os.path.join(os.path.dirname(os.path.abspath(fcidump_path)), "dets.npz")
    if os.path.exists(_dets_path):
        _ref_data      = np.load(_dets_path)
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

    # ---- Precompute owner map and local index map ----
    fragment_orbs_list = [f[0] for f in valid]
    owner      = _compute_owner_map(fragment_orbs_list, n_orb, order)
    local_idx_map = {}
    for fi, frag_orbs in enumerate(fragment_orbs_list):
        for li, r in enumerate(frag_orbs):
            local_idx_map[(fi, r)] = li

    # Log owner assignment for overlap orbitals
    for r in range(n_orb):
        covered_by = [fi for fi, fo in enumerate(fragment_orbs_list) if r in fo]
        if len(covered_by) > 1:
            print(f"  Overlap orb {r:2d}: fragments={covered_by}  owner=F{owner[r]}")

    # ---- Optional out-of-band bare reference solve ----
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

    # ---- Initialize gamma_mixed from reference determinant (embedded Iter 0) ----
    gamma_ref_vec = np.array(
        [((ref_alpha_bits >> r) & 1) + ((ref_beta_bits >> r) & 1)
         for r in range(n_orb)], dtype=np.float64)
    gamma_mixed = gamma_ref_vec.copy()

    # ---- Iteration loop ----
    iteration_history = []
    prev_energies     = None
    prev_gamma_global = None
    converged         = False
    delta_E           = float('inf')
    delta_rdm         = float('inf')
    final_results     = None
    delta_rdm_window  = []

    try:
        for it in range(max_iterations + 1):
            frag_results             = []
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
                    frag_results.append(best_res)
                    fragment_rdm1s_for_gamma.append(gamma_sum / n_gamma_avg)

            # Owner-fragment assembly (v5 key change)
            gamma_new = _assemble_global_rdm1_owner_v5(
                fragment_rdm1s_for_gamma, fragment_orbs_list, n_orb, n_elec_total,
                ref_alpha_bits, ref_beta_bits, owner, local_idx_map)

            # Damping
            gamma_mixed = damping * gamma_new + (1.0 - damping) * gamma_mixed

            # Convergence deltas
            energies = [r.energy for r in frag_results]
            n_dets   = [r.n_dets  for r in frag_results]
            if prev_energies is not None:
                delta_E = max(abs(a - b) for a, b in zip(energies, prev_energies))
            if prev_gamma_global is not None:
                delta_rdm = float(np.max(np.abs(gamma_mixed - prev_gamma_global)))

            # Rolling 10-iter std-dev
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
