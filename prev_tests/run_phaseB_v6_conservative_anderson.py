"""
Run Phase B v6 conservative Anderson diagnostic.

This is a versioned experiment runner. It writes only to a new output folder:
TrimCI_Flow/Outputs/outs_v6_conservative_anderson_avg5_200dets/
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np

from TrimCI_Flow.trimci_flow_v6 import run_selfconsistent_fragments_v6
from TrimCI_Flow.analysis import convergence_summary


ROOT = Path(__file__).resolve().parents[1]
FCIDUMP = ROOT / "Fe4S4_251230orbital_-327.1920_10kdets" / (
    "Fe4S4_251230orbital_-327.1920_10kdets"
) / "fcidump_cycle_6"
OUTDIR = ROOT / "TrimCI_Flow" / "Outputs" / "outs_v6_conservative_anderson_avg5_200dets"

TRIMCI_CONFIG = {
    "threshold": 0.03,
    "max_final_dets": 200,
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


def _finite(values):
    return [float(v) for v in values if not (math.isnan(float(v)) or math.isinf(float(v)))]


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


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    result = run_selfconsistent_fragments_v6(
        str(FCIDUMP),
        window_size=15,
        stride=10,
        max_iterations=60,
        convergence=1e-6,
        rdm_convergence=1e-4,
        damping=0.1,
        anderson_beta=0.1,
        anderson_m=5,
        anderson_reg=1e-2,
        n_gamma_avg=5,
        trimci_config=TRIMCI_CONFIG,
        log_bare_reference=True,
    )

    conv = convergence_summary(result)
    history = list(result.iteration_history)
    anderson_count = sum(1 for row in history if row.get("anderson_used"))
    fallback_count = len(history) - anderson_count

    metadata = {
        "run_type": "Phase_B_v6_conservative_anderson_avg5",
        "timestamp": datetime.now().isoformat(),
        "fcidump": str(FCIDUMP),
        "window_size": 15,
        "stride": 10,
        "max_iterations": 60,
        "convergence": 1e-6,
        "rdm_convergence": 1e-4,
        "damping": 0.1,
        "anderson_beta": 0.1,
        "anderson_m": 5,
        "anderson_reg": 1e-2,
        "fallback_threshold": 0.05,
        "n_gamma_avg": 5,
        "trimci_config": TRIMCI_CONFIG,
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
    _write_iteration_table(history, OUTDIR / "iteration_table.txt")
    np.save(OUTDIR / "gamma_mixed_final.npy", result._gamma_mixed_final)

    print(f"\nSaved to {OUTDIR.relative_to(ROOT)}/")
    print(f"  converged={result.converged}")
    print(f"  iterations={result.iterations}")
    print(f"  final max|dE|={result.convergence_delta:.4e} Ha")
    print(f"  final max|dgamma|={result.convergence_delta_rdm:.4e}")
    print(f"  fragment_n_dets={result.fragment_n_dets}")
    print(f"  Anderson used={anderson_count}/{len(history)}")


if __name__ == "__main__":
    main()
