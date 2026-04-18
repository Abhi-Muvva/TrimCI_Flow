"""
meanfield/solver.py
===================
Self-consistent fragment solve with mean-field (Fock) coupling between fragments.

Algorithm
---------
Each fragment's one-body Hamiltonian is dressed with the mean-field contribution
from all other fragments' 1-RDMs. The global γ vector (spin-summed orbital
occupations) is assembled by overlap-averaging and renormalized to conserve
electron number. We iterate until γ and fragment energies stop changing.

Mixer: Conservative Anderson acceleration (Type II, Walker-Ni) with linear
damping fallback. Parameters tuned so Anderson operates at the same scale
as the stable linear mixer (beta=0.1, reg=1e-2). History is trimmed on
fallback to prevent bad extrapolation from poisoning future steps.

Current best parameters (Fe4S4, W=15, S=10):
    n_gamma_avg   = 5       (external gamma averaging, 5 independent solves)
    damping       = 0.1     (linear fallback coefficient)
    anderson_beta = 0.1     (Anderson blending factor)
    anderson_reg  = 1e-2    (Tikhonov regularization)
    convergence   = 5e-2 Ha (energy, stochastic noise floor)
    rdm_conv      = 1e-2    (max|Δγ|, practical noise floor at N=5)
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np

from TrimCI_Flow.core.results import FragmentedRunResult
from TrimCI_Flow.core.fragment import (
    fragment_by_sliding_window,
    extract_fragment_integrals,
    fragment_electron_count,
)
from TrimCI_Flow.core.trimci_adapter import solve_fragment_trimci
from TrimCI_Flow.meanfield.helpers import (
    compute_fragment_rdm1,
    dress_integrals_meanfield,
    assemble_global_rdm1_diag,
)

# Anderson fallback: revert to linear damping if proposed step exceeds this
_FALLBACK_THRESHOLD = 0.05


def _anderson_step(
    x_hist: List[np.ndarray],
    f_hist: List[np.ndarray],
    m: int,
    beta: float,
    reg: float,
    linear_alpha: float,
) -> Tuple[np.ndarray, bool, float]:
    """
    Anderson Type II (Walker-Ni) mixing step.

    Returns (gamma_next, anderson_used, residual_ratio).
    Falls back to plain linear mixing when history is insufficient or
    when the least-squares solve fails.
    """
    x_hist = x_hist[-(m + 1):]
    f_hist = f_hist[-(m + 1):]
    n_hist = len(x_hist)

    x_k = x_hist[-1]
    f_k = f_hist[-1]
    r_k = f_k - x_k

    if n_hist < 2:
        return x_k + linear_alpha * r_k, False, float('nan')

    n_diff = n_hist - 1
    n = len(x_k)
    DF = np.empty((n, n_diff))
    DX = np.empty((n, n_diff))
    for j in range(n_diff):
        DF[:, j] = f_hist[j + 1] - f_hist[j]
        DX[:, j] = x_hist[j + 1] - x_hist[j]

    A = DF.T @ DF + reg * np.eye(n_diff)
    b = DF.T @ r_k
    try:
        theta = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return x_k + linear_alpha * r_k, False, float('nan')

    gamma_next = x_k + beta * r_k - (DX + beta * DF) @ theta

    r_approx  = r_k - DF @ theta
    r_k_inf   = float(np.max(np.abs(r_k)))
    res_ratio = float(np.max(np.abs(r_approx))) / max(r_k_inf, 1e-15)

    return gamma_next, True, res_ratio


def run_selfconsistent_fragments(
    fcidump_path: str,
    window_size: int = 15,
    stride: int = 10,
    max_iterations: int = 60,
    convergence: float = 5e-2,
    rdm_convergence: float = 1e-2,
    damping: float = 0.1,
    anderson_beta: float = 0.1,
    anderson_m: int = 5,
    anderson_reg: float = 1e-2,
    n_gamma_avg: int = 5,
    trimci_config: Optional[dict] = None,
    log_bare_reference: bool = True,
) -> FragmentedRunResult:
    """
    Self-consistent fragment solve with mean-field Fock coupling.

    Parameters
    ----------
    fcidump_path    : path to FCIDUMP file
    window_size     : orbitals per fragment (default 15)
    stride          : sliding window stride (default 10)
    max_iterations  : maximum SCF cycles (default 60)
    convergence     : energy convergence per fragment in Ha (default 5e-2)
    rdm_convergence : max|Δγ| convergence threshold (default 1e-2)
    damping         : linear mixing fallback coefficient (default 0.1)
    anderson_beta   : Anderson blending factor (default 0.1)
    anderson_m      : Anderson history depth (default 5)
    anderson_reg    : Tikhonov regularization for Anderson (default 1e-2)
    n_gamma_avg     : independent TrimCI solves averaged per fragment per iter
    trimci_config   : optional TrimCI config overrides
    log_bare_reference : run a bare-h1 solve once at start for comparison

    Convergence criteria
    --------------------
    Both max|ΔE_frag| < convergence AND max|Δγ| < rdm_convergence must hold
    simultaneously. Energy is stochastic-noise dominated at 200 dets; the
    gamma criterion is the primary physical convergence metric.

    Notes
    -----
    - γ is initialized from the reference determinant (embedded Iter 0).
      This eliminates the large first-iteration shock from a cold start.
    - Overlap orbitals are averaged across fragments (not owner-assigned).
      This provides a real sqrt(2) noise reduction for shared orbitals.
    - h1 is always dressed from the bare integrals each iteration (no drift).
    - Electron count per fragment is frozen at initialization.
    """
    import trimci

    h1, eri, n_elec, n_orb, E_nuc, n_alpha, n_beta, psym = trimci.read_fcidump(fcidump_path)
    order = np.argsort(np.diag(h1))

    _dets_path = os.path.join(os.path.dirname(os.path.abspath(fcidump_path)), "dets.npz")
    if os.path.exists(_dets_path):
        _ref = np.load(_dets_path)
        ref_alpha_bits = int(_ref["dets"][0, 0])
        ref_beta_bits  = int(_ref["dets"][0, 1])
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

    # Optional bare-reference solve (comparison baseline)
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

    # Initialize γ from reference determinant (embedded Iter 0)
    gamma_ref_vec = np.array(
        [((ref_alpha_bits >> r) & 1) + ((ref_beta_bits >> r) & 1)
         for r in range(n_orb)], dtype=np.float64)
    gamma_mixed = gamma_ref_vec.copy()

    # Anderson state
    anderson_x_hist: List[np.ndarray] = []
    anderson_f_hist: List[np.ndarray] = []

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

            fragment_orbs_list = [f[0] for f in valid]
            gamma_new = assemble_global_rdm1_diag(
                fragment_rdm1s_for_gamma, fragment_orbs_list, n_orb, n_elec_total,
                ref_alpha_bits, ref_beta_bits)

            anderson_x_hist.append(gamma_mixed.copy())
            anderson_f_hist.append(gamma_new.copy())

            gamma_proposed, anderson_used, residual_ratio = _anderson_step(
                anderson_x_hist, anderson_f_hist, anderson_m, anderson_beta,
                anderson_reg, damping)

            gamma_proposed = np.clip(gamma_proposed, 0.0, 2.0)

            max_proposed_change = float(np.max(np.abs(gamma_proposed - gamma_mixed)))
            if anderson_used and max_proposed_change > _FALLBACK_THRESHOLD:
                gamma_proposed = damping * gamma_new + (1.0 - damping) * gamma_mixed
                anderson_used  = False
                residual_ratio = float('nan')
                anderson_x_hist = anderson_x_hist[-2:]
                anderson_f_hist = anderson_f_hist[-2:]
                print(f"  Iter {it}: Anderson fallback (step={max_proposed_change:.3f})")

            gamma_mixed = gamma_proposed

            energies = [r.energy for r in frag_results]
            n_dets   = [r.n_dets  for r in frag_results]
            if prev_energies is not None:
                delta_E = max(abs(a - b) for a, b in zip(energies, prev_energies))
            if prev_gamma_global is not None:
                delta_rdm = float(np.max(np.abs(gamma_mixed - prev_gamma_global)))

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
                "anderson_used":   anderson_used,
                "residual_ratio":  residual_ratio,
            })
            a_flag = 'A' if anderson_used else 'L'
            print(f"  Iter {it}: max|dE|={delta_E:.2e}, max|dgamma|={delta_rdm:.2e}, "
                  f"rdm_std10={rdm_rolling_std:.2e}, ndets={sum(n_dets)}, mix={a_flag}")

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
    result._bare_reference   = bare_reference
    result._gamma_mixed_final = gamma_mixed
    result._ref_alpha_bits    = ref_alpha_bits
    result._ref_beta_bits     = ref_beta_bits
    return result
