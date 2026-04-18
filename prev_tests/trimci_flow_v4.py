"""
trimci_flow_v4.py
=================
Phase B improvements over trimci_flow_v3.py:
  1. Anderson Type II (Walker-Ni) acceleration replaces linear damping as the
     primary mixer.  Fallback to linear damping (alpha=damping) when the
     Anderson-proposed step is too large (> FALLBACK_THRESHOLD) or when
     history is insufficient (< 2 pairs).
  2. n_gamma_avg=5 default — best-confirmed noise level from Run 7.
  3. anderson_beta=0.5 as the Anderson blending factor (5x more aggressive than
     the fallback alpha=0.1, but still stabilized by the history fit).
  4. anderson_reg=1e-4 — Tikhonov regularization relative to typical DF
     singular-value scale (||DF||_spectral ~ 0.1), set to suppress directions
     with singular value below ~3% of the signal (noise level ~0.007/sigma).
  5. History cleared on fallback to keep history consistent (no mixed
     Anderson/linear splice artefacts).

Scientific rationale:
  At N=5, Run 7 reached rolling std10(max|Δgamma|) ≈ 0.007 while the
  systematic fixed-point residual is ~0.016–0.022 (SNR ≈ 2–3:1).  Anderson
  extrapolates along the systematic residual direction using the last m pairs
  (gamma_in, gamma_out), allowing super-linear contraction when the residual
  is approximately predictable.  The regularization damps noise-dominated
  directions in the DF matrix.  With beta=0.5 inside Anderson, the update is
  more aggressive than linear alpha=0.1, while the fallback to alpha=0.1
  prevents divergence if Anderson overshoots.

Path C (run_fragmented_trimci) unchanged, imported from trimci_flow.
"""
from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple

from TrimCI_Flow.trimci_flow import (
    FragmentedRunResult,
    run_fragmented_trimci,       # noqa: F401 — re-exported
    compute_fragment_rdm1,
    dress_integrals_meanfield,
)
from TrimCI_Flow.trimci_flow_v2 import _assemble_global_rdm1_diag_v2


# If Anderson proposes a step with max|Δgamma| above this, revert to linear
FALLBACK_THRESHOLD = 0.30


def _anderson_step(
    x_hist: List[np.ndarray],
    f_hist: List[np.ndarray],
    m: int,
    beta: float,
    reg: float,
) -> Tuple[np.ndarray, bool, float]:
    """
    Anderson Type II (Walker-Ni) mixing.

    Parameters
    ----------
    x_hist : recent gamma_in  vectors (most recent last, length >= 1)
    f_hist : recent gamma_out = F(gamma_in) vectors, same length
    m      : history depth   (use at most m+1 pairs)
    beta   : Anderson blending factor
    reg    : Tikhonov regularization strength

    Returns
    -------
    gamma_next    : proposed next iterate (not yet clipped)
    anderson_used : True if >= 2 history pairs were available
    residual_ratio: ||r_k - DF theta||_inf / ||r_k||_inf  (quality of fit;
                    nan when history < 2 or linear fallback used)
    """
    # Limit to most recent m+1 entries
    x_hist = x_hist[-(m + 1):]
    f_hist = f_hist[-(m + 1):]
    n_hist = len(x_hist)

    x_k = x_hist[-1]
    f_k = f_hist[-1]
    r_k = f_k - x_k  # current residual

    if n_hist < 2:
        # Not enough history — plain linear mixing
        return x_k + beta * r_k, False, float('nan')

    n_diff = n_hist - 1
    n = len(x_k)

    # Build finite-difference matrices
    DF = np.empty((n, n_diff))
    DX = np.empty((n, n_diff))
    for j in range(n_diff):
        DF[:, j] = f_hist[j + 1] - f_hist[j]
        DX[:, j] = x_hist[j + 1] - x_hist[j]

    # Tikhonov-regularized least squares: min ||r_k - DF θ||² + reg ||θ||²
    A = DF.T @ DF + reg * np.eye(n_diff)
    b = DF.T @ r_k
    try:
        theta = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return x_k + beta * r_k, False, float('nan')

    # Anderson update
    gamma_next = x_k + beta * r_k - (DX + beta * DF) @ theta

    # Residual-fit quality: how much of r_k was explained by DF
    r_approx   = r_k - DF @ theta
    r_k_inf    = float(np.max(np.abs(r_k)))
    res_ratio  = float(np.max(np.abs(r_approx))) / max(r_k_inf, 1e-15)

    return gamma_next, True, res_ratio


def run_selfconsistent_fragments_v4(
    fcidump_path: str,
    window_size: int,
    stride: int,
    max_iterations: int = 60,
    convergence: float = 1e-6,
    rdm_convergence: float = 1e-4,
    damping: float = 0.1,        # fallback linear mixing coefficient
    anderson_beta: float = 0.5,  # Anderson blending factor
    anderson_m: int = 5,         # Anderson history depth
    anderson_reg: float = 1e-4,  # Tikhonov regularization
    n_gamma_avg: int = 5,
    trimci_config: Optional[dict] = None,
    log_bare_reference: bool = True,
) -> FragmentedRunResult:
    """
    Self-consistent fragment solve with Anderson-accelerated mixing (Path B), v4.

    Parameters
    ----------
    fcidump_path    : path to FCIDUMP file
    window_size     : orbitals per fragment
    stride          : sliding window stride
    max_iterations  : maximum SCF cycles (default 60)
    convergence     : energy convergence threshold per fragment (Ha)
    rdm_convergence : max|Delta gamma| convergence threshold
    damping         : fallback linear mixing coefficient when Anderson reverts
    anderson_beta   : Anderson blending factor (0 < beta <= 1)
    anderson_m      : Anderson history depth (default 5)
    anderson_reg    : Tikhonov regularization strength (default 1e-4)
    n_gamma_avg     : independent TrimCI solves averaged per fragment per iter
    trimci_config   : optional TrimCI config overrides
    log_bare_reference : run bare-h1 solve out-of-band and log (Path C ref)

    Differences from v3
    -------------------
    - Anderson Type II replaces linear damping as primary mixer.
    - Fallback to linear damping (alpha=damping) if Anderson proposes
      max|Δgamma| > FALLBACK_THRESHOLD=0.30; history is cleared on fallback.
    - anderson_beta=0.5 (5x more aggressive than fallback alpha=0.1).
    - anderson_reg=1e-4 (suppresses noise-dominated DF directions).
    - n_gamma_avg=5 default.
    - iteration_history includes 'anderson_used' and 'residual_ratio' fields.
    """
    import trimci
    import os
    from TrimCI_Flow.fragment import (
        fragment_by_sliding_window,
        extract_fragment_integrals,
        fragment_electron_count,
    )
    from TrimCI_Flow.trimci_adapter import solve_fragment_trimci

    # ---- Setup (same as v3) ----
    h1, eri, n_elec, n_orb, E_nuc, n_alpha, n_beta, psym = trimci.read_fcidump(fcidump_path)
    order = np.argsort(np.diag(h1))

    _dets_path = os.path.join(os.path.dirname(os.path.abspath(fcidump_path)), "dets.npz")
    if os.path.exists(_dets_path):
        _ref_data  = np.load(_dets_path)
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

    # ---- Anderson state ----
    anderson_x_hist: List[np.ndarray] = []   # gamma_in  per iteration
    anderson_f_hist: List[np.ndarray] = []   # gamma_out per iteration

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
                    frag_results.append(best_res)
                    fragment_rdm1s_for_gamma.append(gamma_sum / n_gamma_avg)

            fragment_orbs_list = [f[0] for f in valid]
            gamma_new = _assemble_global_rdm1_diag_v2(
                fragment_rdm1s_for_gamma, fragment_orbs_list, n_orb, n_elec_total,
                ref_alpha_bits, ref_beta_bits)

            # Accumulate history for Anderson (current pair appended before calling)
            anderson_x_hist.append(gamma_mixed.copy())
            anderson_f_hist.append(gamma_new.copy())

            # Anderson step
            gamma_proposed, anderson_used, residual_ratio = _anderson_step(
                anderson_x_hist, anderson_f_hist, anderson_m, anderson_beta, anderson_reg)

            # Clip to physical range
            gamma_proposed = np.clip(gamma_proposed, 0.0, 2.0)

            # Stability fallback: revert to linear damping if step is too large
            max_proposed_change = float(np.max(np.abs(gamma_proposed - gamma_mixed)))
            if anderson_used and max_proposed_change > FALLBACK_THRESHOLD:
                gamma_proposed = damping * gamma_new + (1.0 - damping) * gamma_mixed
                anderson_used  = False
                residual_ratio = float('nan')
                # Clear history to avoid corrupting future Anderson steps
                anderson_x_hist = anderson_x_hist[-2:]
                anderson_f_hist = anderson_f_hist[-2:]
                print(f"  Iter {it}: Anderson fallback (step={max_proposed_change:.3f})")

            gamma_mixed = gamma_proposed

            # Convergence deltas
            energies = [r.energy for r in frag_results]
            n_dets   = [r.n_dets  for r in frag_results]
            if prev_energies is not None:
                delta_E = max(abs(a - b) for a, b in zip(energies, prev_energies))
            if prev_gamma_global is not None:
                delta_rdm = float(np.max(np.abs(gamma_mixed - prev_gamma_global)))

            # Rolling 10-iter std-dev of max|Δgamma|
            if delta_rdm != float('inf'):
                delta_rdm_window.append(delta_rdm)
                if len(delta_rdm_window) > 10:
                    delta_rdm_window.pop(0)
            rdm_rolling_std = (float(np.std(delta_rdm_window))
                               if len(delta_rdm_window) >= 3 else float('nan'))

            iteration_history.append({
                "iteration":      it,
                "energies":       energies,
                "n_dets":         n_dets,
                "delta_E":        delta_E,
                "delta_rdm":      delta_rdm,
                "rdm_rolling_std": rdm_rolling_std,
                "anderson_used":  anderson_used,
                "residual_ratio": residual_ratio,
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
    result._bare_reference = bare_reference
    return result
