"""
Meanfield SCF + extraction run.

Answers the core Phase B scientific question:
does the converged mean-field dressed Hamiltonian reduce total determinants
relative to the uncoupled baseline of 118?

Steps
-----
1. Run meanfield SCF to convergence (relaxed criteria)
2. Apply the converged γ_mixed to each fragment's dressed h1
3. Solve each fragment with Phase-C-equivalent settings (max_final_dets="auto",
   threshold=0.06) to get a fair determinant-count comparison

Result (Fe4S4, W=15, S=10, N=5):
    extraction fragment_n_dets: [51, 51, 16] = 118 total
    uncoupled baseline:                         118 total
    verdict: Phase B is neutral — same determinant count as uncoupled

Output: TrimCI_Flow/Outputs/meanfield_active/outs_extraction_autodets/
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np

from TrimCI_Flow.meanfield.solver import run_selfconsistent_fragments
from TrimCI_Flow.meanfield.helpers import dress_integrals_meanfield
from TrimCI_Flow.core.fragment import (
    fragment_by_sliding_window,
    extract_fragment_integrals,
    fragment_electron_count,
)
from TrimCI_Flow.core.trimci_adapter import solve_fragment_trimci
from TrimCI_Flow.core.analysis import convergence_summary

ROOT    = Path(__file__).resolve().parents[3]
FCIDUMP = ROOT / "Fe4S4_251230orbital_-327.1920_10kdets" / (
    "Fe4S4_251230orbital_-327.1920_10kdets"
) / "fcidump_cycle_6"
OUTDIR  = ROOT / "TrimCI_Flow" / "Outputs" / "meanfield_active" / "outs_extraction_autodets"

UNCOUPLED_BASELINE_DETS = 118

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


def _jsonable(v):
    if isinstance(v, np.ndarray): return v.tolist()
    if isinstance(v, np.generic):  return v.item()
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)): return v
    if isinstance(v, float):       return float(v)
    if isinstance(v, dict):        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [_jsonable(x) for x in v]
    return v


def _window_stats(history, start, end):
    vals  = [float(r["delta_rdm"]) for r in history
             if start <= r["iteration"] <= end and not math.isinf(float(r["delta_rdm"]))]
    evals = [float(r["delta_E"])   for r in history
             if start <= r["iteration"] <= end and not math.isinf(float(r["delta_E"]))]
    if not vals: return {}
    s = {"delta_rdm_min": float(np.min(vals)), "delta_rdm_median": float(np.median(vals)),
         "delta_rdm_max": float(np.max(vals))}
    if evals:
        s.update({"delta_E_min": float(np.min(evals)), "delta_E_median": float(np.median(evals)),
                  "delta_E_max": float(np.max(evals))})
    return s


def _threshold_stats(history, thresholds):
    out = {}
    for t in thresholds:
        hits = [r["iteration"] for r in history
                if not math.isinf(float(r["delta_rdm"])) and float(r["delta_rdm"]) < t]
        out[str(t)] = {"count": len(hits), "first_iter": hits[0] if hits else None}
    return out


def _write_iteration_table(history, path):
    lines = [
        " iter | max|dE| Ha | max|dgamma| | rdm_std10 | mix | res_ratio | total_dets",
        "------|------------|-------------|-----------|-----|-----------|-----------",
    ]
    for row in history:
        d_e = float(row["delta_E"]); d_g = float(row["delta_rdm"])
        std = float(row.get("rdm_rolling_std", float("nan")))
        res = float(row.get("residual_ratio", float("nan")))
        mix = "A" if row.get("anderson_used") else "L"
        lines.append(
            f"{int(row['iteration']):5d} | "
            f"{'inf' if math.isinf(d_e) else f'{d_e:.3e}':>10} | "
            f"{'inf' if math.isinf(d_g) else f'{d_g:.3e}':>11} | "
            f"{'nan' if math.isnan(std) else f'{std:.3e}':>9} | {mix:^3} | "
            f"{'nan' if math.isnan(res) else f'{res:.3e}':>9} | {sum(row['n_dets']):9d}"
        )
    path.write_text("\n".join(lines) + "\n")


def _run_extraction(gamma_mixed, ref_alpha_bits, ref_beta_bits):
    import trimci
    h1, eri, _n_elec, n_orb, _e_nuc, _n_alpha, _n_beta, _psym = trimci.read_fcidump(str(FCIDUMP))
    order = np.argsort(np.diag(h1))
    valid = []
    for frag_orbs in fragment_by_sliding_window(n_orb, order, 15, 10):
        na, nb = fragment_electron_count(ref_alpha_bits, ref_beta_bits, frag_orbs)
        n_frag = len(frag_orbs)
        if na == 0 or nb == 0 or na > n_frag or nb > n_frag:
            continue
        h1_f, eri_f = extract_fragment_integrals(h1, eri, frag_orbs)
        valid.append((frag_orbs, na, nb, h1_f, eri_f))

    extraction = []
    for idx, (frag_orbs, na, nb, h1_bare, eri_f) in enumerate(valid):
        ext_orbs  = [r for r in range(n_orb) if r not in set(frag_orbs)]
        ext_gamma = gamma_mixed[np.asarray(ext_orbs, dtype=np.intp)]
        h1_use    = dress_integrals_meanfield(h1_bare, eri, frag_orbs, ext_gamma, ext_orbs)
        res = solve_fragment_trimci(h1_use, eri_f, na, nb, len(frag_orbs), EXTRACTION_CONFIG)
        extraction.append({
            "fragment": idx, "orbs": list(map(int, frag_orbs)),
            "n_alpha": int(na), "n_beta": int(nb),
            "energy": float(res.energy), "n_dets": int(res.n_dets),
        })
        print(f"  [extract] F{idx}: n_dets={res.n_dets}, energy={res.energy:.8f} Ha")
    return extraction


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if (OUTDIR / "run_metadata.json").exists():
        raise FileExistsError(f"{OUTDIR} already contains completed output — delete to re-run")

    result = run_selfconsistent_fragments(
        str(FCIDUMP),
        window_size=15, stride=10,
        max_iterations=60,
        convergence=5e-2, rdm_convergence=1e-2,
        damping=0.1, anderson_beta=0.1, anderson_m=5, anderson_reg=1e-2,
        n_gamma_avg=5, trimci_config=SCF_CONFIG, log_bare_reference=True,
    )

    conv         = convergence_summary(result)
    history      = list(result.iteration_history)
    gamma_mixed  = result._gamma_mixed_final
    a_count      = sum(1 for r in history if r.get("anderson_used"))

    np.save(OUTDIR / "gamma_mixed_final.npy", gamma_mixed)

    print("\nRunning extraction on converged dressed Hamiltonians...")
    extraction = _run_extraction(gamma_mixed, int(result._ref_alpha_bits), int(result._ref_beta_bits))
    total_dets = int(sum(r["n_dets"] for r in extraction))
    if total_dets < UNCOUPLED_BASELINE_DETS:
        verdict = "Meanfield reduces determinant count vs uncoupled"
    elif total_dets == UNCOUPLED_BASELINE_DETS:
        verdict = "Meanfield is neutral: same determinant count as uncoupled"
    else:
        verdict = "Meanfield increases determinant count vs uncoupled"

    metadata = {
        "run_type": "meanfield_extraction_autodets",
        "timestamp": datetime.now().isoformat(),
        "fcidump": str(FCIDUMP),
        "scf": {
            "window_size": 15, "stride": 10, "max_iterations": 60,
            "convergence": 5e-2, "rdm_convergence": 1e-2,
            "damping": 0.1, "anderson_beta": 0.1, "anderson_m": 5, "anderson_reg": 1e-2,
            "fallback_threshold": 0.05, "n_gamma_avg": 5,
            "trimci_config": SCF_CONFIG,
            "converged": result.converged, "iterations_performed": result.iterations,
            "total_dets_final": result.total_dets, "fragment_n_dets_final": result.fragment_n_dets,
            "fragment_energies_final": result.fragment_energies,
            "convergence_delta": result.convergence_delta,
            "convergence_delta_rdm": result.convergence_delta_rdm,
            "bare_reference": getattr(result, "_bare_reference", None),
            "late_stats_40_60": _window_stats(history, 40, 60),
            "late_stats_50_60": _window_stats(history, 50, 60),
            "threshold_dgamma": _threshold_stats(history, [0.05, 0.02, 0.01, 0.005]),
            "anderson_count": a_count, "linear_or_fallback_count": len(history) - a_count,
        },
        "extraction": {
            "trimci_config": EXTRACTION_CONFIG,
            "fragment_results": extraction,
            "fragment_n_dets": [r["n_dets"] for r in extraction],
            "total_dets": total_dets,
            "uncoupled_baseline_total_dets": UNCOUPLED_BASELINE_DETS,
            "delta_vs_uncoupled": total_dets - UNCOUPLED_BASELINE_DETS,
            "verdict": verdict,
        },
    }

    (OUTDIR / "run_metadata.json").write_text(json.dumps(_jsonable(metadata), indent=2, allow_nan=True) + "\n")
    (OUTDIR / "iteration_history.json").write_text(json.dumps(_jsonable(history), indent=2, allow_nan=True) + "\n")
    (OUTDIR / "convergence_summary.json").write_text(json.dumps(_jsonable(conv), indent=2, allow_nan=True) + "\n")
    (OUTDIR / "extraction_results.json").write_text(json.dumps(_jsonable(metadata["extraction"]), indent=2) + "\n")
    _write_iteration_table(history, OUTDIR / "iteration_table.txt")

    print(f"\nSaved to {OUTDIR.relative_to(ROOT)}/")
    print(f"  SCF converged={result.converged}, iters={result.iterations}")
    print(f"  extraction fragment_n_dets={[r['n_dets'] for r in extraction]}")
    print(f"  extraction total_dets={total_dets}  (uncoupled baseline={UNCOUPLED_BASELINE_DETS})")
    print(f"  verdict: {verdict}")


if __name__ == "__main__":
    main()
