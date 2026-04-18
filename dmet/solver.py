# TrimCI_Flow/dmet/solver.py
"""
solver.py
=========
1-shot non-overlapping DMET orchestration.

Fragment partition: 3 non-overlapping groups of n_orb//3 orbitals,
sorted by h1 diagonal energy (same ordering as Phase C / meanfield).

Pipeline per fragment:
  1. schmidt_decomp(gamma_mf, frag_orbs, n_orb)
  2. build_impurity_hamiltonian(h1, eri, gamma_core_full, env_orbs, P_imp)
  3. impurity_electron_count(n_elec_total, n_elec_core)
  4. solve_fragment_trimci(h1_solver, eri_proj, n_alpha_imp, n_beta_imp, n_imp)
  5. rdm_energy_check(...) for every fragment
  6. compute and record diagnostics
  7. dmet_energy_a/b/c per fragment

E_DMET = E_nuc + sum(E_I_b for I in fragments)
"""
from __future__ import annotations

import os
import json
import numpy as np
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


def run_dmet_1shot(
    fcidump_path: str,
    trimci_config: Optional[dict] = None,
    output_dir: Optional[str] = None,
) -> FragmentedRunResult:
    """
    1-shot non-overlapping DMET with TrimCI as impurity solver.

    Parameters
    ----------
    fcidump_path  : path to FCIDUMP file (9.6 MB Fe4S4 file)
    trimci_config : optional TrimCI config overrides (passed to solve_fragment_trimci)
    output_dir    : if provided, writes results.json here

    Returns
    -------
    FragmentedRunResult with runtime attributes:
      .E_dmet        : float, canonical DMET energy from formula B (Ha)
      .E_dmet_a      : float, formula A energy (debug)
      .E_dmet_c      : float, formula C energy (diagnostic)
      .E_hf          : float, RHF baseline energy
    """
    import trimci as _trimci

    # ── 1. Read FCIDUMP ───────────────────────────────────────────────────────
    h1, eri, n_elec, n_orb, E_nuc, n_alpha, n_beta, psym = _trimci.read_fcidump(fcidump_path)
    print(f"  [DMET] FCIDUMP: n_orb={n_orb}, n_elec={n_elec}, E_nuc={E_nuc}")

    # ── 2. Fragment partition (non-overlapping, h1-diagonal ordering) ─────────
    order = np.argsort(np.diag(h1))       # same ordering as Phase C/B
    n_frag = n_orb // 3                   # = 12 for Fe4S4
    fragments = [
        sorted(order[0        : n_frag].tolist()),
        sorted(order[n_frag   : 2*n_frag].tolist()),
        sorted(order[2*n_frag : n_orb].tolist()),
    ]
    print(f"  [DMET] Fragments: F0={fragments[0][:3]}..., F1={fragments[1][:3]}..., F2={fragments[2][:3]}...")

    # ── 3. RHF reference ─────────────────────────────────────────────────────
    gamma_mf, e_hf = run_hf(h1, eri, n_elec, n_orb)
    print(f"  [DMET] E_HF = {e_hf:.6f} Ha  (Tr(gamma_mf)={np.trace(gamma_mf):.4f})")

    # ── 4. Per-fragment impurity solve ────────────────────────────────────────
    fragment_energies   = []
    fragment_n_dets     = []
    energies_a          = []
    energies_b          = []
    energies_c          = []
    fragment_diagnostics = []

    for frag_idx, frag_orbs in enumerate(fragments):
        print(f"\n  [DMET] === Fragment {frag_idx} ({len(frag_orbs)} orbs) ===")

        # Schmidt decomp + bath
        P_imp, gamma_core_full, n_elec_core, n_bath, env_orbs = schmidt_decomp(
            gamma_mf, frag_orbs, n_orb)
        n_imp = len(frag_orbs) + n_bath
        print(f"    n_bath={n_bath}, n_core={len(env_orbs)-n_bath}, n_elec_core={n_elec_core:.3f}")

        # Electron count for impurity
        n_alpha_imp, n_beta_imp = impurity_electron_count(n_elec, n_elec_core)
        print(f"    n_elec_imp={n_alpha_imp+n_beta_imp} (alpha={n_alpha_imp}, beta={n_beta_imp})")

        # Impurity Hamiltonian
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
        print(f"    E_imp={result.energy:.6f} Ha, n_dets={result.n_dets}")
        fragment_energies.append(result.energy)
        fragment_n_dets.append(result.n_dets)

        # RDM extraction + 2-RDM convention check for every fragment.
        gamma, gamma2, e_rdm, rdm_discrepancy = rdm_energy_check(
            result.dets, result.coeffs, n_imp,
            h1_solver, eri_proj, result.energy)
        if rdm_discrepancy > 1e-6:
            raise RuntimeError(
                f"2-RDM convention check FAILED on fragment {frag_idx}: "
                f"energy_from_rdm={e_rdm:.8f} Ha, "
                f"TrimCI reported={result.energy:.8f} Ha, "
                f"discrepancy={rdm_discrepancy:.2e} Ha (tol=1e-6). "
                f"Check h1_solver (must include Fock correction) and eri_proj convention."
            )
        print(f"  [DMET] 2-RDM convention check PASSED on fragment {frag_idx} "
              f"(discrepancy={rdm_discrepancy:.2e} Ha)")

        # Energy formulas
        nfrag = len(frag_orbs)
        ea = dmet_energy_a(h1_phys_proj, gamma, nfrag)
        eb = dmet_energy_b(h1_solver,    eri_proj, gamma, gamma2, nfrag)
        ec = dmet_energy_c(h1_solver,    eri_proj, gamma, gamma2, nfrag)
        print(f"    E_A={ea:.6f}  E_B={eb:.6f}  E_C={ec:.6f}")
        energies_a.append(ea); energies_b.append(eb); energies_c.append(ec)

        gamma_trace = float(np.trace(gamma))
        n_elec_imp = n_alpha_imp + n_beta_imp
        eri_sym_pq = float(np.max(np.abs(eri_proj - eri_proj.transpose(1, 0, 2, 3))))
        eri_sym_rs = float(np.max(np.abs(eri_proj - eri_proj.transpose(0, 1, 3, 2))))
        eri_sym_pair = float(np.max(np.abs(eri_proj - eri_proj.transpose(2, 3, 0, 1))))
        fragment_diagnostics.append({
            "fragment_index": frag_idx,
            "n_frag": nfrag,
            "n_bath": n_bath,
            "n_imp": n_imp,
            "n_core": len(env_orbs) - n_bath,
            "n_elec_core": float(n_elec_core),
            "n_alpha_imp": int(n_alpha_imp),
            "n_beta_imp": int(n_beta_imp),
            "n_elec_imp": int(n_elec_imp),
            "impurity_trimci_energy": float(result.energy),
            "rdm_reconstructed_impurity_energy": float(e_rdm),
            "rdm_reconstructed_energy_discrepancy": float(rdm_discrepancy),
            "E_A": float(ea),
            "E_B": float(eb),
            "E_C": float(ec),
            "gamma_trace": gamma_trace,
            "gamma_min": float(np.min(gamma)),
            "gamma_max": float(np.max(gamma)),
            "gamma_trace_error_vs_impurity_electrons": float(gamma_trace - n_elec_imp),
            "gamma_trace_matches_impurity_electrons": bool(np.isclose(gamma_trace, n_elec_imp, atol=1e-6)),
            "h1_solver_max_symmetry_violation": float(np.max(np.abs(h1_solver - h1_solver.T))),
            "eri_proj_max_symmetry_violation_pq": eri_sym_pq,
            "eri_proj_max_symmetry_violation_rs": eri_sym_rs,
            "eri_proj_max_symmetry_violation_pair": eri_sym_pair,
            "v_fock_correction_norm": float(np.linalg.norm(v_fock_imp)),
            "v_fock_correction_max_abs": float(np.max(np.abs(v_fock_imp))),
        })

    # ── 5. Total energies ─────────────────────────────────────────────────────
    E_dmet_a = E_nuc + sum(energies_a)
    E_dmet_b = E_nuc + sum(energies_b)
    E_dmet_c = E_nuc + sum(energies_c)

    print(f"\n{'='*60}")
    print(f"=== DMET 1-shot (non-overlapping 12+12+12) ===")
    print(f"  E_HF                           = {e_hf:.6f} Ha")
    print(f"  E_DMET_A (1-body, debug)       = {E_dmet_a:.6f} Ha")
    print(f"  E_DMET_B (2-RDM, primary)      = {E_dmet_b:.6f} Ha  <- canonical")
    print(f"  E_DMET_C (democratic, diag.)   = {E_dmet_c:.6f} Ha")
    print(f"  Reference (brute-force TrimCI) = -327.1920 Ha")
    print(f"  Error (B - reference)          = {E_dmet_b - (-327.1920):+.4f} Ha")
    print(f"  Total DMET dets: {sum(fragment_n_dets)}  (Phase C baseline: 118,  brute-force: 10095)")
    print(f"{'='*60}")

    # ── 6. Build result object ────────────────────────────────────────────────
    result_obj = FragmentedRunResult(
        fragment_energies = fragment_energies,
        fragment_n_dets   = fragment_n_dets,
        fragment_orbs     = fragments,
        total_dets        = sum(fragment_n_dets),
        iterations        = 1,
    )
    result_obj.E_dmet       = E_dmet_b
    result_obj.E_dmet_a     = E_dmet_a
    result_obj.E_dmet_c     = E_dmet_c
    result_obj.E_hf         = e_hf

    # ── 7. Optionally write JSON ──────────────────────────────────────────────
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        payload = {
            "E_dmet_b":        E_dmet_b,
            "E_dmet_a":        E_dmet_a,
            "E_dmet_c":        E_dmet_c,
            "E_hf":            e_hf,
            "trimci_config":    trimci_config,
            "fragment_n_dets": fragment_n_dets,
            "total_dets":      sum(fragment_n_dets),
            "fragment_orbs":   fragments,
            "fragment_energies_imp": fragment_energies,
            "fragment_diagnostics": fragment_diagnostics,
        }
        with open(os.path.join(output_dir, "results.json"), "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  [DMET] Results written to {output_dir}/results.json")

    return result_obj
