# TrimCI_Flow/dmet/sc_solver.py
"""
sc_solver.py
============
Self-consistent (SC) DMET with TrimCI as impurity solver.

Algorithm (Knizia & Chan 2012):
  1. Initialize correlation potential u_emb = 0
  2. Run UHF with h1_eff = h1 + u_emb  →  γ_MF
  3. For each fragment: Schmidt decomp → build impurity H → TrimCI solve
     Extract γ_imp_I[frag_I, frag_I] from impurity 1-RDM
  4. Convergence check: max|γ_imp[frag] - γ_MF[frag]| < conv_tol ?
  5. Update u_blocks via direct-fitting step:
       u_I += step * (γ_imp_I - γ_MF[frag_I])  then damp with u_damp
  6. Rebuild u_emb and go to step 2

Convergence guarantees that the bath is self-consistent with the correlated
wavefunction → E_B partition is physically meaningful.

Energies (A, B, C) are computed at every iteration for convergence monitoring.
The 2-RDM convention check gates every B/C contraction.
"""
from __future__ import annotations

import os
import json
import traceback
import numpy as np
from datetime import datetime
from typing import Optional

from TrimCI_Flow.core.results import FragmentedRunResult
from TrimCI_Flow.core.trimci_adapter import solve_fragment_trimci
from TrimCI_Flow.dmet.hf_reference import run_hf
from TrimCI_Flow.dmet.bath import (
    schmidt_decomp,
    build_impurity_hamiltonian,
    impurity_electron_count,
)
from TrimCI_Flow.dmet.energy import (
    dmet_energy_a,
    dmet_energy_b,
    dmet_energy_c,
    rdm_energy_check,
)
from TrimCI_Flow.dmet.correlation_potential import (
    build_u_emb,
    extract_gamma_frag,
    update_u_blocks,
    damp_u_blocks,
    gamma_frag_mismatch,
    u_blocks_max_change,
    zero_u_blocks,
)


def _solve_fragments(
    h1: np.ndarray,
    eri: np.ndarray,
    n_elec: int,
    n_orb: int,
    gamma_mf: np.ndarray,
    fragments: list[list[int]],
    trimci_config: Optional[dict],
    sc_iter: int,
) -> tuple[list, list, list, list, list, list, list]:
    """
    Run Schmidt decomp + TrimCI + RDM extraction for all fragments.

    Returns
    -------
    fragment_energies   : list of TrimCI impurity energies
    fragment_n_dets     : list of determinant counts
    gamma_imp_frags     : list of (n_frag_I, n_frag_I) fragment-block γ_imp
    energies_a/b/c      : per-fragment formula A/B/C energies
    fragment_diagnostics: list of per-fragment diagnostic dicts
    per_frag_bath       : list of n_bath per fragment (for consistency checking)
    """
    fragment_energies    = []
    fragment_n_dets      = []
    gamma_imp_frags      = []
    energies_a           = []
    energies_b           = []
    energies_c           = []
    fragment_diagnostics = []
    per_frag_bath        = []

    for frag_idx, frag_orbs in enumerate(fragments):
        n_frag = len(frag_orbs)
        print(f"    [SC iter {sc_iter}] Fragment {frag_idx} ({n_frag} orbs)")

        # Schmidt decomp
        P_imp, gamma_core_full, n_elec_core, n_bath, env_orbs = schmidt_decomp(
            gamma_mf, frag_orbs, n_orb)
        n_imp = n_frag + n_bath
        per_frag_bath.append(n_bath)
        print(f"      n_bath={n_bath}, n_core={len(env_orbs)-n_bath}, "
              f"n_elec_core={n_elec_core:.3f}")

        # Electron count
        n_alpha_imp, n_beta_imp = impurity_electron_count(n_elec, n_elec_core)
        n_elec_imp = n_alpha_imp + n_beta_imp
        print(f"      n_elec_imp={n_elec_imp} (alpha={n_alpha_imp}, beta={n_beta_imp})")

        # Impurity Hamiltonian (note: h1 here is h1+u_emb passed in via gamma_mf context;
        # the bath/Fock construction uses the PHYSICAL h1 and eri, not h1+u_emb)
        h1_phys_proj, h1_solver, eri_proj = build_impurity_hamiltonian(
            h1, eri, gamma_core_full, env_orbs, P_imp)
        v_fock_imp = h1_solver - h1_phys_proj

        # TrimCI solve
        result = solve_fragment_trimci(
            h1_solver, eri_proj,
            n_alpha_imp, n_beta_imp,
            n_orb_frag=n_imp,
            config=trimci_config,
        )
        print(f"      E_imp={result.energy:.6f} Ha,  n_dets={result.n_dets}")
        fragment_energies.append(result.energy)
        fragment_n_dets.append(result.n_dets)

        # 2-RDM convention check + γ and γ2 extraction
        gamma_full, gamma2, e_rdm, rdm_disc = rdm_energy_check(
            result.dets, result.coeffs, n_imp,
            h1_solver, eri_proj, result.energy)
        if rdm_disc > 1e-6:
            raise RuntimeError(
                f"[SC iter {sc_iter}] 2-RDM convention check FAILED on fragment {frag_idx}: "
                f"energy_from_rdm={e_rdm:.8f} Ha, TrimCI={result.energy:.8f} Ha, "
                f"discrepancy={rdm_disc:.2e} Ha"
            )
        print(f"      2-RDM check PASSED (discrepancy={rdm_disc:.2e} Ha)")

        # Fragment-block of impurity γ (indices 0:n_frag in impurity space = fragment orbitals)
        gamma_imp_frag_I = gamma_full[:n_frag, :n_frag].copy()
        gamma_imp_frags.append(gamma_imp_frag_I)

        # Energy formulas
        ea = dmet_energy_a(h1_phys_proj, gamma_full, n_frag)
        eb = dmet_energy_b(h1_solver,    eri_proj, gamma_full, gamma2, n_frag)
        ec = dmet_energy_c(h1_solver,    eri_proj, gamma_full, gamma2, n_frag)
        print(f"      E_A={ea:.6f}  E_B={eb:.6f}  E_C={ec:.6f}")
        energies_a.append(ea)
        energies_b.append(eb)
        energies_c.append(ec)

        # Diagnostics
        gamma_trace = float(np.trace(gamma_full))
        fragment_diagnostics.append({
            "fragment_index": frag_idx,
            "n_frag": n_frag,
            "n_bath": n_bath,
            "n_imp": n_imp,
            "n_core": len(env_orbs) - n_bath,
            "n_elec_core": float(n_elec_core),
            "n_alpha_imp": int(n_alpha_imp),
            "n_beta_imp": int(n_beta_imp),
            "n_elec_imp": int(n_elec_imp),
            "impurity_trimci_energy": float(result.energy),
            "rdm_reconstructed_energy_discrepancy": float(rdm_disc),
            "E_A": float(ea),
            "E_B": float(eb),
            "E_C": float(ec),
            "gamma_trace": gamma_trace,
            "gamma_trace_matches_impurity_electrons": bool(
                np.isclose(gamma_trace, n_elec_imp, atol=1e-6)),
            "v_fock_correction_norm": float(np.linalg.norm(v_fock_imp)),
        })

    return (fragment_energies, fragment_n_dets, gamma_imp_frags,
            energies_a, energies_b, energies_c,
            fragment_diagnostics, per_frag_bath)


def run_dmet_sc(
    fcidump_path: str,
    trimci_config: Optional[dict] = None,
    output_dir: Optional[str] = None,
    max_sc_iter: int = 30,
    conv_tol: float = 1e-3,
    u_damp: float = 0.5,
    u_step: float = 1.0,
) -> FragmentedRunResult:
    """
    Self-consistent non-overlapping DMET with TrimCI as impurity solver.

    Parameters
    ----------
    fcidump_path  : path to FCIDUMP file (9.6 MB Fe4S4 file)
    trimci_config : optional TrimCI config overrides
    output_dir    : if provided, writes results.json and sc_summary.md here
    max_sc_iter   : maximum SC iterations (default 30)
    conv_tol      : convergence threshold for max|γ_imp[frag] - γ_MF[frag]|
                    (default 1e-3)
    u_damp        : linear mixing fraction for u update in (0, 1].
                    u_mixed = u_damp * u_new + (1-u_damp) * u_old.
                    Default 0.5 (recommended for Fe4S4).
    u_step        : direct-fitting gradient step size (default 1.0)

    Returns
    -------
    FragmentedRunResult with runtime attributes:
      .E_dmet          : float, final canonical DMET energy from formula B (Ha)
      .E_dmet_a        : float, formula A (debug)
      .E_dmet_c        : float, formula C (diagnostic)
      .E_hf            : float, UHF energy at final iteration
      .sc_converged    : bool
      .sc_iterations   : int, number of SC iterations performed
      .iteration_history: list of per-iteration dicts
    """
    import trimci as _trimci

    start_time = datetime.now()
    iteration_history = []
    converged = False

    # ── 1. Read FCIDUMP ───────────────────────────────────────────────────────
    h1, eri, n_elec, n_orb, E_nuc, n_alpha, n_beta, psym = _trimci.read_fcidump(fcidump_path)
    print(f"[SC-DMET] FCIDUMP: n_orb={n_orb}, n_elec={n_elec}, E_nuc={E_nuc}")

    # ── 2. Fragment partition (non-overlapping, h1-diagonal ordering) ─────────
    order  = np.argsort(np.diag(h1))
    n_frag = n_orb // 3      # = 12 for Fe4S4
    fragments = [
        sorted(order[0      : n_frag].tolist()),
        sorted(order[n_frag : 2 * n_frag].tolist()),
        sorted(order[2 * n_frag : n_orb].tolist()),
    ]
    print(f"[SC-DMET] Fragments: F0={fragments[0][:3]}..., "
          f"F1={fragments[1][:3]}..., F2={fragments[2][:3]}...")
    print(f"[SC-DMET] SC params: max_iter={max_sc_iter}, conv_tol={conv_tol}, "
          f"u_damp={u_damp}, u_step={u_step}")

    # ── 3. Initialize correlation potential ───────────────────────────────────
    u_blocks = zero_u_blocks(fragments)
    u_emb    = build_u_emb(u_blocks, fragments, n_orb)   # all zeros initially

    # Keep track of previous u for Δu monitoring
    u_blocks_prev = [u.copy() for u in u_blocks]

    # ── 4. SC loop ────────────────────────────────────────────────────────────
    final_fragment_energies  = []
    final_fragment_n_dets    = []
    final_energies_a         = []
    final_energies_b         = []
    final_energies_c         = []
    final_fragment_diag      = []
    final_gamma_mf           = None
    final_e_hf               = None
    prev_n_bath              = None

    try:
        for sc_iter in range(max_sc_iter):
            print(f"\n{'='*64}")
            print(f"[SC-DMET] Iteration {sc_iter}")
            print(f"{'='*64}")

            # 4a. UHF with h1 + u_emb
            gamma_mf, e_hf = run_hf(h1 + u_emb, eri, n_elec, n_orb)
            print(f"  E_HF = {e_hf:.6f} Ha  (Tr(γ_MF)={np.trace(gamma_mf):.4f})")

            # 4b. Per-fragment impurity solves
            (frag_energies, frag_n_dets, gamma_imp_frags,
             energies_a, energies_b, energies_c,
             frag_diag, per_frag_bath) = _solve_fragments(
                h1, eri, n_elec, n_orb, gamma_mf, fragments,
                trimci_config, sc_iter)

            # n_bath consistency check across iterations
            if prev_n_bath is not None and per_frag_bath != prev_n_bath:
                print(f"  [SC-DMET WARNING] n_bath changed: {prev_n_bath} → {per_frag_bath}. "
                      f"Bath rank shifted due to u_emb change.")
            prev_n_bath = per_frag_bath

            # 4c. Convergence metric: max|γ_imp[frag] - γ_MF[frag]|
            per_frag_delta, global_max_delta = gamma_frag_mismatch(
                gamma_mf, gamma_imp_frags, fragments)

            # Δu metric (secondary)
            delta_u = u_blocks_max_change(u_blocks, u_blocks_prev)

            # Energy summary for this iteration
            E_b_iter = E_nuc + sum(energies_b)
            E_a_iter = E_nuc + sum(energies_a)
            E_c_iter = E_nuc + sum(energies_c)

            print(f"\n  [SC iter {sc_iter}] Convergence:  "
                  f"max|Δγ_frag|={global_max_delta:.4e}  (tol={conv_tol:.1e})  "
                  f"max|Δu|={delta_u:.4e}")
            print(f"  [SC iter {sc_iter}] E_B={E_b_iter:.6f} Ha  "
                  f"(ref -327.1920, err={E_b_iter+327.1920:+.4f} Ha)")
            print(f"  [SC iter {sc_iter}] per-frag Δγ: {[f'{d:.4e}' for d in per_frag_delta]}")

            # n_bath sanity check — should stay at n_frag (=12) throughout
            n_frag_size = len(fragments[0])
            if any(nb < n_frag_size for nb in per_frag_bath):
                print(f"  [SC-DMET WARNING] n_bath collapsed: {per_frag_bath} "
                      f"(expected all == {n_frag_size}). "
                      f"UHF may have converged to a closed-shell solution. "
                      f"Consider reducing u_step or u_damp.")

            # Record iteration
            iteration_history.append({
                "sc_iter": sc_iter,
                "E_hf": float(e_hf),
                "E_dmet_b": float(E_b_iter),
                "E_dmet_a": float(E_a_iter),
                "E_dmet_c": float(E_c_iter),
                "max_gamma_frag_delta": float(global_max_delta),
                "per_frag_gamma_delta": [float(d) for d in per_frag_delta],
                "max_u_delta": float(delta_u),
                "fragment_n_dets": list(frag_n_dets),
                "total_dets": sum(frag_n_dets),
                "per_frag_n_bath": list(per_frag_bath),
                "converged": False,
            })

            # Save final-iteration data in case we converge or hit max_iter
            final_fragment_energies = frag_energies
            final_fragment_n_dets   = frag_n_dets
            final_energies_a        = energies_a
            final_energies_b        = energies_b
            final_energies_c        = energies_c
            final_fragment_diag     = frag_diag
            final_gamma_mf          = gamma_mf
            final_e_hf              = e_hf

            # 4d. Convergence check
            if global_max_delta < conv_tol:
                converged = True
                iteration_history[-1]["converged"] = True
                print(f"\n[SC-DMET] CONVERGED at iteration {sc_iter}  "
                      f"max|Δγ_frag|={global_max_delta:.4e} < {conv_tol:.1e}")
                break

            # 4e. Update u: direct-fitting step then damp
            u_blocks_prev = [u.copy() for u in u_blocks]
            u_blocks_new  = update_u_blocks(
                u_blocks, gamma_mf, gamma_imp_frags, fragments, step=u_step)
            u_blocks      = damp_u_blocks(u_blocks_new, u_blocks, alpha=u_damp)
            u_emb         = build_u_emb(u_blocks, fragments, n_orb)

        else:
            # max_sc_iter exhausted without convergence
            print(f"\n[SC-DMET] NOT CONVERGED after {max_sc_iter} iterations. "
                  f"Final max|Δγ_frag|={iteration_history[-1]['max_gamma_frag_delta']:.4e}")

    except Exception:
        tb = traceback.format_exc()
        print(f"\n[SC-DMET] EXCEPTION in SC loop:\n{tb}")
        # Write failure record before re-raising
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            failure_payload = {
                "status": "FAILURE",
                "traceback": tb,
                "iteration_history": iteration_history,
            }
            with open(os.path.join(output_dir, "results.json"), "w") as f:
                json.dump(failure_payload, f, indent=2)
        raise

    # ── 5. Final energy summary ───────────────────────────────────────────────
    sc_iters_done = len(iteration_history)
    E_dmet_a = E_nuc + sum(final_energies_a)
    E_dmet_b = E_nuc + sum(final_energies_b)
    E_dmet_c = E_nuc + sum(final_energies_c)

    print(f"\n{'='*64}")
    print(f"=== SC-DMET Final Result ({sc_iters_done} iterations) ===")
    print(f"  Converged              : {converged}")
    print(f"  E_HF  (last iter)      = {final_e_hf:.6f} Ha")
    print(f"  E_DMET_A (1-body, dbg) = {E_dmet_a:.6f} Ha")
    print(f"  E_DMET_B (2-RDM, pri)  = {E_dmet_b:.6f} Ha  <- canonical")
    print(f"  E_DMET_C (democratic)  = {E_dmet_c:.6f} Ha")
    print(f"  Reference (brute-force)= -327.1920    Ha")
    print(f"  Error (B - reference)  = {E_dmet_b - (-327.1920):+.4f} Ha")
    print(f"  Total DMET dets        : {sum(final_fragment_n_dets)}")
    print(f"{'='*64}")

    # ── 6. Build result object ────────────────────────────────────────────────
    result_obj = FragmentedRunResult(
        fragment_energies = final_fragment_energies,
        fragment_n_dets   = final_fragment_n_dets,
        fragment_orbs     = fragments,
        total_dets        = sum(final_fragment_n_dets),
        iterations        = sc_iters_done,
    )
    result_obj.E_dmet          = E_dmet_b
    result_obj.E_dmet_a        = E_dmet_a
    result_obj.E_dmet_c        = E_dmet_c
    result_obj.E_hf            = final_e_hf
    result_obj.sc_converged    = converged
    result_obj.sc_iterations   = sc_iters_done
    result_obj.iteration_history = iteration_history

    # ── 7. Write output files ─────────────────────────────────────────────────
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # results.json — full record
        payload = {
            "status": "CONVERGED" if converged else "NOT_CONVERGED",
            "sc_iterations": sc_iters_done,
            "sc_params": {
                "max_sc_iter": max_sc_iter,
                "conv_tol": conv_tol,
                "u_damp": u_damp,
                "u_step": u_step,
            },
            "trimci_config": trimci_config,
            "E_dmet_b": float(E_dmet_b),
            "E_dmet_a": float(E_dmet_a),
            "E_dmet_c": float(E_dmet_c),
            "E_hf": float(final_e_hf),
            "error_vs_reference": float(E_dmet_b - (-327.1920)),
            "fragment_n_dets": final_fragment_n_dets,
            "total_dets": sum(final_fragment_n_dets),
            "fragment_orbs": fragments,
            "fragment_energies_imp": final_fragment_energies,
            "fragment_diagnostics": final_fragment_diag,
            "iteration_history": iteration_history,
        }
        results_path = os.path.join(output_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[SC-DMET] results.json → {results_path}")

        # sc_summary.md — human-readable, ready to paste into progress.md
        elapsed = (datetime.now() - start_time).total_seconds()
        _write_sc_summary(output_dir, payload, elapsed, fcidump_path, trimci_config)

    return result_obj


def _write_sc_summary(
    output_dir: str,
    payload: dict,
    elapsed_sec: float,
    fcidump_path: str,
    trimci_config: Optional[dict],
) -> None:
    """Write sc_summary.md — a progress.md-ready section for this SC-DMET run."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    status   = payload["status"]
    sc_iters = payload["sc_iterations"]
    params   = payload["sc_params"]
    tc       = trimci_config or {}

    hist = payload["iteration_history"]
    # Build convergence table (every iteration)
    table_rows = []
    for row in hist:
        n_bath_str = str(row.get('per_frag_n_bath', '?'))
        table_rows.append(
            f"| {row['sc_iter']:3d} | {row['E_dmet_b']:15.6f} | "
            f"{row['max_gamma_frag_delta']:.4e} | "
            f"{row['max_u_delta']:.4e} | "
            f"{row['total_dets']:5d} | "
            f"{n_bath_str} | "
            f"{'✓' if row['converged'] else ''} |"
        )

    lines = [
        f"# DMET SC — {ts}",
        "",
        f"**Status:** {status}  |  **SC iterations:** {sc_iters}  |  "
        f"**Elapsed:** {elapsed_sec/60:.1f} min",
        "",
        "## Parameters",
        "",
        f"- `max_sc_iter` = {params['max_sc_iter']}",
        f"- `conv_tol`    = {params['conv_tol']}",
        f"- `u_damp`      = {params['u_damp']}",
        f"- `u_step`      = {params['u_step']}",
        f"- TrimCI `threshold`      = {tc.get('threshold', 'N/A')}",
        f"- TrimCI `max_final_dets` = {tc.get('max_final_dets', 'N/A')}",
        f"- TrimCI `max_rounds`     = {tc.get('max_rounds', 'N/A')}",
        "",
        "## Final energies",
        "",
        f"| Quantity | Value (Ha) |",
        f"|----------|-----------|",
        f"| E_HF (UHF, last iter) | {payload['E_hf']:.6f} |",
        f"| E_DMET_A (1-body, debug) | {payload['E_dmet_a']:.6f} |",
        f"| **E_DMET_B (2-RDM, primary)** | **{payload['E_dmet_b']:.6f}** |",
        f"| E_DMET_C (democratic) | {payload['E_dmet_c']:.6f} |",
        f"| Error vs −327.1920 | {payload['error_vs_reference']:+.4f} |",
        "",
        f"Total dets: {payload['total_dets']}  "
        f"(fragments: {payload['fragment_n_dets']})",
        "",
        "## SC convergence history",
        "",
        "| iter | E_DMET_B (Ha) | max\\|Δγ_frag\\| | max\\|Δu\\| | total dets | n_bath | conv |",
        "|-----:|-------------:|---------------:|----------:|----------:|-------|------|",
    ] + table_rows + [
        "",
        "## Fragment diagnostics (final iteration)",
        "",
    ]

    for d in payload["fragment_diagnostics"]:
        lines += [
            f"- **F{d['fragment_index']}**: n_imp={d['n_imp']} "
            f"(frag={d['n_frag']}, bath={d['n_bath']}, core={d['n_core']}), "
            f"n_elec_imp={d['n_elec_imp']}, "
            f"E_imp={d['impurity_trimci_energy']:.6f} Ha, "
            f"E_B={d['E_B']:.6f} Ha, "
            f"RDM disc={d['rdm_reconstructed_energy_discrepancy']:.2e} Ha",
        ]

    lines += ["", "---", ""]

    summary_path = os.path.join(output_dir, "sc_summary.md")
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[SC-DMET] sc_summary.md → {summary_path}")
