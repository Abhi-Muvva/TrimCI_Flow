# TrimCI_Flow/dmet/runners/run_dmet_sc.py
"""
Runner: Self-consistent DMET on Fe4S4 with TrimCI as impurity solver.

Usage (from /home/unfunnypanda/Proj_Flow/):
    source qflowenv/bin/activate

    # Quick test (~5-10 min, loose thresholds):
    python TrimCI_Flow/dmet/runners/run_dmet_sc.py --preset quick 2>&1 | tee sc_quick.log

    # Standard run (~30-60 min per iteration × 30 iterations):
    python TrimCI_Flow/dmet/runners/run_dmet_sc.py --preset standard 2>&1 | tee sc_standard.log

    # Custom override (all flags optional):
    python TrimCI_Flow/dmet/runners/run_dmet_sc.py \\
        --threshold 0.02 --max-final-dets 500 \\
        --max-sc-iter 20 --conv-tol 1e-3 --u-damp 0.5 2>&1 | tee sc_custom.log

Presets
-------
quick    : threshold=0.06, max_final_dets="auto" (~50 dets), max_sc_iter=15
           Use this first to verify SC loop runs without errors.

standard : threshold=0.02, max_final_dets=500, max_sc_iter=30
           Use this for production runs after quick test passes.

Outputs written to TrimCI_Flow/Outputs/dmet/outs_dmet_sc_<preset>_<timestamp>/
  - results.json   : full SC history + final energies
  - sc_summary.md  : human-readable progress.md section
"""
import sys
import os
import argparse
from datetime import datetime

# ── project root on path ─────────────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..')))

FCIDUMP = (
    "/home/unfunnypanda/Proj_Flow/"
    "Fe4S4_251230orbital_-327.1920_10kdets/"
    "Fe4S4_251230orbital_-327.1920_10kdets/"
    "fcidump_cycle_6"
)

PRESETS = {
    "quick": {
        "trimci": {
            "threshold":           0.06,
            "max_final_dets":      "auto",
            "max_rounds":          2,
            "num_runs":            1,
            "pool_build_strategy": "heat_bath",
            "verbose":             False,
        },
        "max_sc_iter": 15,
        "conv_tol":    1e-3,
        "u_damp":      0.5,
        "u_step":      0.1,   # conservative: sign was wrong in v1; 0.1 safe starting point
    },
    "standard": {
        "trimci": {
            "threshold":           0.02,
            "max_final_dets":      500,
            "max_rounds":          2,
            "num_runs":            1,
            "pool_build_strategy": "heat_bath",
            "verbose":             False,
        },
        "max_sc_iter": 30,
        "conv_tol":    1e-3,
        "u_damp":      0.5,
        "u_step":      0.1,   # conservative; increase to 0.5 if convergence is confirmed
    },
}


def parse_args():
    p = argparse.ArgumentParser(description="SC-DMET runner for Fe4S4")
    p.add_argument("--preset",         default="standard",
                   choices=list(PRESETS.keys()),
                   help="Configuration preset (default: standard)")
    p.add_argument("--threshold",      type=float, default=None,
                   help="TrimCI threshold override")
    p.add_argument("--max-final-dets", type=int,   default=None,
                   dest="max_final_dets",
                   help="TrimCI max_final_dets override")
    p.add_argument("--max-sc-iter",    type=int,   default=None,
                   dest="max_sc_iter",
                   help="Max SC iterations override")
    p.add_argument("--conv-tol",       type=float, default=None,
                   dest="conv_tol",
                   help="SC convergence tolerance override (default 1e-3)")
    p.add_argument("--u-damp",         type=float, default=None,
                   dest="u_damp",
                   help="Correlation potential damping factor override (default 0.5)")
    p.add_argument("--u-step",         type=float, default=None,
                   dest="u_step",
                   help="Direct-fitting gradient step size override (default 1.0)")
    return p.parse_args()


def main():
    args = parse_args()

    # Start from preset, then apply CLI overrides
    preset = PRESETS[args.preset]
    trimci_config = dict(preset["trimci"])
    max_sc_iter   = preset["max_sc_iter"]
    conv_tol      = preset["conv_tol"]
    u_damp        = preset["u_damp"]
    u_step        = preset["u_step"]

    if args.threshold      is not None: trimci_config["threshold"]      = args.threshold
    if args.max_final_dets is not None: trimci_config["max_final_dets"] = args.max_final_dets
    if args.max_sc_iter    is not None: max_sc_iter = args.max_sc_iter
    if args.conv_tol       is not None: conv_tol    = args.conv_tol
    if args.u_damp         is not None: u_damp      = args.u_damp
    if args.u_step         is not None: u_step      = args.u_step

    # Output directory
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..', 'Outputs', 'dmet',
        f"outs_dmet_sc_{args.preset}_{timestamp}"))
    os.makedirs(output_dir, exist_ok=True)

    print(f"[runner] SC-DMET  preset={args.preset}")
    print(f"[runner] TrimCI config : {trimci_config}")
    print(f"[runner] SC params     : max_iter={max_sc_iter}, conv_tol={conv_tol}, "
          f"u_damp={u_damp}, u_step={u_step}")
    print(f"[runner] Output dir    : {output_dir}")
    print()

    from TrimCI_Flow.dmet.sc_solver import run_dmet_sc

    result = run_dmet_sc(
        fcidump_path = FCIDUMP,
        trimci_config = trimci_config,
        output_dir    = output_dir,
        max_sc_iter   = max_sc_iter,
        conv_tol      = conv_tol,
        u_damp        = u_damp,
        u_step        = u_step,
    )

    print(f"\n[runner] Done.")
    print(f"[runner] SC converged      : {result.sc_converged}")
    print(f"[runner] SC iterations     : {result.sc_iterations}")
    print(f"[runner] E_DMET_B (final)  : {result.E_dmet:.6f} Ha")
    print(f"[runner] Error vs -327.1920: {result.E_dmet + 327.1920:+.4f} Ha")
    print(f"[runner] Total dets        : {result.total_dets}")
    print(f"[runner] Output dir        : {output_dir}")


if __name__ == "__main__":
    main()
