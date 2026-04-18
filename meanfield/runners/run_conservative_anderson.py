"""
Meanfield SCF — conservative Anderson, strict convergence criteria.

Strict criteria (1e-6 Ha, 1e-4 gamma) are not met in practice due to
stochastic CI noise, but this runner documents the raw algorithm behavior.
Use run_relaxed.py for a converging run.

Output: TrimCI_Flow/Outputs/meanfield_active/outs_conservative_anderson_n5/
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np

from TrimCI_Flow.meanfield.solver import run_selfconsistent_fragments
from TrimCI_Flow.core.analysis import convergence_summary


ROOT   = Path(__file__).resolve().parents[3]
FCIDUMP = ROOT / "Fe4S4_251230orbital_-327.1920_10kdets" / (
    "Fe4S4_251230orbital_-327.1920_10kdets"
) / "fcidump_cycle_6"
OUTDIR = ROOT / "TrimCI_Flow" / "Outputs" / "meanfield_active" / "outs_conservative_anderson_n5"

TRIMCI_CONFIG = {
    "threshold": 0.03,
    "max_final_dets": 200,
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


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    result = run_selfconsistent_fragments(
        str(FCIDUMP),
        window_size=15, stride=10,
        max_iterations=60,
        convergence=1e-6, rdm_convergence=1e-4,
        damping=0.1, anderson_beta=0.1, anderson_m=5, anderson_reg=1e-2,
        n_gamma_avg=5, trimci_config=TRIMCI_CONFIG, log_bare_reference=True,
    )

    conv    = convergence_summary(result)
    history = list(result.iteration_history)
    a_count = sum(1 for r in history if r.get("anderson_used"))

    metadata = {
        "run_type": "meanfield_conservative_anderson_n5",
        "timestamp": datetime.now().isoformat(),
        "fcidump": str(FCIDUMP),
        "window_size": 15, "stride": 10, "max_iterations": 60,
        "convergence": 1e-6, "rdm_convergence": 1e-4,
        "damping": 0.1, "anderson_beta": 0.1, "anderson_m": 5, "anderson_reg": 1e-2,
        "fallback_threshold": 0.05, "n_gamma_avg": 5,
        "trimci_config": TRIMCI_CONFIG,
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
    }

    (OUTDIR / "run_metadata.json").write_text(json.dumps(_jsonable(metadata), indent=2, allow_nan=True) + "\n")
    (OUTDIR / "iteration_history.json").write_text(json.dumps(_jsonable(history), indent=2, allow_nan=True) + "\n")
    (OUTDIR / "convergence_summary.json").write_text(json.dumps(_jsonable(conv), indent=2, allow_nan=True) + "\n")
    _write_iteration_table(history, OUTDIR / "iteration_table.txt")
    np.save(OUTDIR / "gamma_mixed_final.npy", result._gamma_mixed_final)

    print(f"\nSaved to {OUTDIR.relative_to(ROOT)}/")
    print(f"  converged={result.converged}, iters={result.iterations}")
    print(f"  final max|dE|={result.convergence_delta:.4e} Ha")
    print(f"  final max|dgamma|={result.convergence_delta_rdm:.4e}")
    print(f"  Anderson used={a_count}/{len(history)}")


if __name__ == "__main__":
    main()
