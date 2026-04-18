"""
Run Phase B v6 SCF and extract Phase-C-style determinant counts.

This runner answers the Phase B determinant question:
does the converged mean-field dressed Hamiltonian reduce total determinants
relative to the Phase C baseline of 118?

It writes only to:
TrimCI_Flow/Outputs/outs_v6_extraction_autodets_threshold006/
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np

from TrimCI_Flow.analysis import convergence_summary
from TrimCI_Flow.fragment import (
    extract_fragment_integrals,
    fragment_by_sliding_window,
    fragment_electron_count,
)
from TrimCI_Flow.trimci_adapter import solve_fragment_trimci
from TrimCI_Flow.trimci_flow import dress_integrals_meanfield
from TrimCI_Flow.trimci_flow_v6 import run_selfconsistent_fragments_v6


ROOT = Path(__file__).resolve().parents[1]
FCIDUMP = ROOT / "Fe4S4_251230orbital_-327.1920_10kdets" / (
    "Fe4S4_251230orbital_-327.1920_10kdets"
) / "fcidump_cycle_6"
OUTDIR = ROOT / "TrimCI_Flow" / "Outputs" / "outs_v6_extraction_autodets_threshold006"
PHASE_C_BASELINE_DETS = 118

SCF_CONFIG = {
    "threshold": 0.03,
    "max_final_dets": 200,
    "max_rounds": 2,
    "num_runs": 1,
    "pool_build_strategy": "heat_bath",
    "verbose": False,
}

EXTRACTION_CONFIG = {
    "threshold": 0.06,
    "max_final_dets": "auto",
    "max_rounds": 2,
    "num_runs": 1,
    "pool_build_strategy": "heat_bath",
    "verbose": False,
}


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return value
        return float(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _window_stats(history, start, end):
    vals = [
        float(row["delta_rdm"])
        for row in history
        if start <= int(row["iteration"]) <= end
        and not math.isinf(float(row["delta_rdm"]))
    ]
    evals = [
        float(row["delta_E"])
        for row in history
        if start <= int(row["iteration"]) <= end
        and not math.isinf(float(row["delta_E"]))
    ]
    if not vals:
        return {}
    stats = {
        "delta_rdm_min": float(np.min(vals)),
        "delta_rdm_median": float(np.median(vals)),
        "delta_rdm_max": float(np.max(vals)),
    }
    if evals:
        stats.update({
            "delta_E_min": float(np.min(evals)),
            "delta_E_median": float(np.median(evals)),
            "delta_E_max": float(np.max(evals)),
        })
    return stats


def _threshold_stats(history, thresholds):
    out = {}
    for threshold in thresholds:
        hits = [
            int(row["iteration"])
            for row in history
            if not math.isinf(float(row["delta_rdm"]))
            and float(row["delta_rdm"]) < threshold
        ]
        out[str(threshold)] = {
            "count": len(hits),
            "first_iter": hits[0] if hits else None,
        }
    return out


def _write_iteration_table(history, path):
    lines = [
        " iter | max|dE| Ha | max|dgamma| | rdm_std10 | mix | res_ratio | total_dets",
        "------|------------|-------------|-----------|-----|-----------|-----------",
    ]
    for row in history:
        d_e = float(row["delta_E"])
        d_g = float(row["delta_rdm"])
        std = float(row.get("rdm_rolling_std", float("nan")))
        res = float(row.get("residual_ratio", float("nan")))
        d_e_s = "inf" if math.isinf(d_e) else f"{d_e:.3e}"
        d_g_s = "inf" if math.isinf(d_g) else f"{d_g:.3e}"
        std_s = "nan" if math.isnan(std) else f"{std:.3e}"
        res_s = "nan" if math.isnan(res) else f"{res:.3e}"
        mix = "A" if row.get("anderson_used") else "L"
        lines.append(
            f"{int(row['iteration']):5d} | {d_e_s:>10} | {d_g_s:>11} | "
            f"{std_s:>9} | {mix:^3} | {res_s:>9} | {sum(row['n_dets']):9d}"
        )
    path.write_text("\n".join(lines) + "\n")


def _valid_fragments(fcidump_path, ref_alpha_bits, ref_beta_bits, window_size, stride):
    import trimci

    h1, eri, _n_elec, n_orb, _e_nuc, _n_alpha, _n_beta, _psym = trimci.read_fcidump(
        str(fcidump_path)
    )
    order = np.argsort(np.diag(h1))
    valid = []
    for frag_orbs in fragment_by_sliding_window(n_orb, order, window_size, stride):
        na, nb = fragment_electron_count(ref_alpha_bits, ref_beta_bits, frag_orbs)
        n_frag = len(frag_orbs)
        if na == 0 or nb == 0 or na > n_frag or nb > n_frag:
            continue
        h1_f, eri_f = extract_fragment_integrals(h1, eri, frag_orbs)
        valid.append((frag_orbs, na, nb, h1_f, eri_f))
    return h1, eri, n_orb, valid


def _run_extraction(gamma_mixed, ref_alpha_bits, ref_beta_bits):
    h1, eri, n_orb, valid = _valid_fragments(
        FCIDUMP,
        ref_alpha_bits,
        ref_beta_bits,
        window_size=15,
        stride=10,
    )

    extraction = []
    for idx, (frag_orbs, na, nb, h1_bare, eri_f) in enumerate(valid):
        frag_set = set(frag_orbs)
        ext_orbs = [r for r in range(n_orb) if r not in frag_set]
        ext_gamma = gamma_mixed[np.asarray(ext_orbs, dtype=np.intp)]
        h1_use = dress_integrals_meanfield(h1_bare, eri, frag_orbs, ext_gamma, ext_orbs)
        res = solve_fragment_trimci(
            h1_use,
            eri_f,
            na,
            nb,
            len(frag_orbs),
            EXTRACTION_CONFIG,
        )
        extraction.append({
            "fragment": idx,
            "orbs": list(map(int, frag_orbs)),
            "n_alpha": int(na),
            "n_beta": int(nb),
            "energy": float(res.energy),
            "n_dets": int(res.n_dets),
        })
        print(
            f"  [extract] F{idx}: n_dets={res.n_dets}, "
            f"energy={res.energy:.8f} Ha"
        )
    return extraction


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if (OUTDIR / "run_metadata.json").exists():
        raise FileExistsError(
            f"{OUTDIR} already contains completed extraction output"
        )

    result = run_selfconsistent_fragments_v6(
        str(FCIDUMP),
        window_size=15,
        stride=10,
        max_iterations=60,
        convergence=5e-2,
        rdm_convergence=1e-2,
        damping=0.1,
        anderson_beta=0.1,
        anderson_m=5,
        anderson_reg=1e-2,
        n_gamma_avg=5,
        trimci_config=SCF_CONFIG,
        log_bare_reference=True,
    )

    conv = convergence_summary(result)
    history = list(result.iteration_history)
    gamma_mixed = result._gamma_mixed_final
    np.save(OUTDIR / "gamma_mixed_final.npy", gamma_mixed)

    print("\nRunning Phase-C-style extraction on converged dressed Hamiltonians...")
    extraction = _run_extraction(
        gamma_mixed,
        int(result._ref_alpha_bits),
        int(result._ref_beta_bits),
    )
    extraction_total_dets = int(sum(row["n_dets"] for row in extraction))
    if extraction_total_dets < PHASE_C_BASELINE_DETS:
        verdict = "Phase B succeeds: fewer determinants than Phase C"
    elif extraction_total_dets == PHASE_C_BASELINE_DETS:
        verdict = "Phase B is neutral: same determinant count as Phase C"
    else:
        verdict = "Phase B increases determinant count vs Phase C"

    anderson_count = sum(1 for row in history if row.get("anderson_used"))
    fallback_count = len(history) - anderson_count

    metadata = {
        "run_type": "Phase_B_v6_extraction_autodets_threshold006",
        "timestamp": datetime.now().isoformat(),
        "fcidump": str(FCIDUMP),
        "scf": {
            "window_size": 15,
            "stride": 10,
            "max_iterations": 60,
            "convergence": 5e-2,
            "rdm_convergence": 1e-2,
            "damping": 0.1,
            "anderson_beta": 0.1,
            "anderson_m": 5,
            "anderson_reg": 1e-2,
            "fallback_threshold": 0.05,
            "n_gamma_avg": 5,
            "trimci_config": SCF_CONFIG,
            "converged": result.converged,
            "iterations_performed": result.iterations,
            "total_dets_final": result.total_dets,
            "fragment_n_dets_final": result.fragment_n_dets,
            "fragment_energies_final": result.fragment_energies,
            "convergence_delta": result.convergence_delta,
            "convergence_delta_rdm": result.convergence_delta_rdm,
            "bare_reference": getattr(result, "_bare_reference", None),
            "late_stats_40_60": _window_stats(history, 40, 60),
            "late_stats_50_60": _window_stats(history, 50, 60),
            "threshold_dgamma": _threshold_stats(history, [0.05, 0.02, 0.01, 0.005]),
            "anderson_count": anderson_count,
            "linear_or_fallback_count": fallback_count,
        },
        "extraction": {
            "trimci_config": EXTRACTION_CONFIG,
            "fragment_results": extraction,
            "fragment_n_dets": [row["n_dets"] for row in extraction],
            "total_dets": extraction_total_dets,
            "phaseC_baseline_total_dets": PHASE_C_BASELINE_DETS,
            "delta_vs_phaseC": extraction_total_dets - PHASE_C_BASELINE_DETS,
            "verdict": verdict,
        },
    }

    (OUTDIR / "run_metadata.json").write_text(
        json.dumps(_jsonable(metadata), indent=2, allow_nan=True) + "\n"
    )
    (OUTDIR / "iteration_history.json").write_text(
        json.dumps(_jsonable(history), indent=2, allow_nan=True) + "\n"
    )
    (OUTDIR / "convergence_summary.json").write_text(
        json.dumps(_jsonable(conv), indent=2, allow_nan=True) + "\n"
    )
    (OUTDIR / "extraction_results.json").write_text(
        json.dumps(_jsonable(metadata["extraction"]), indent=2) + "\n"
    )
    _write_iteration_table(history, OUTDIR / "iteration_table.txt")

    lines = [
        "Phase B v6 Extraction Summary",
        "=============================",
        f"SCF converged: {result.converged}",
        f"SCF iterations: {result.iterations}",
        f"SCF final max|Delta E|: {result.convergence_delta:.6e} Ha",
        f"SCF final max|Delta gamma|: {result.convergence_delta_rdm:.6e}",
        "",
        "Extraction config:",
        json.dumps(EXTRACTION_CONFIG, indent=2),
        "",
        "Extraction determinant counts:",
    ]
    for row in extraction:
        lines.append(
            f"  Fragment {row['fragment']}: n_dets={row['n_dets']}, "
            f"energy={row['energy']:.8f} Ha"
        )
    lines.extend([
        f"  Total extraction dets: {extraction_total_dets}",
        f"  Phase C baseline dets: {PHASE_C_BASELINE_DETS}",
        f"  Delta vs Phase C: {extraction_total_dets - PHASE_C_BASELINE_DETS:+d}",
        f"  Verdict: {verdict}",
    ])
    (OUTDIR / "extraction_summary.txt").write_text("\n".join(lines) + "\n")

    print(f"\nSaved to {OUTDIR.relative_to(ROOT)}/")
    print(f"  SCF converged={result.converged}")
    print(f"  SCF iterations={result.iterations}")
    print(f"  SCF final max|dE|={result.convergence_delta:.4e} Ha")
    print(f"  SCF final max|dgamma|={result.convergence_delta_rdm:.4e}")
    print(f"  extraction fragment_n_dets={[row['n_dets'] for row in extraction]}")
    print(f"  extraction total_dets={extraction_total_dets}")
    print(f"  phaseC baseline={PHASE_C_BASELINE_DETS}")
    print(f"  verdict={verdict}")


if __name__ == "__main__":
    main()
