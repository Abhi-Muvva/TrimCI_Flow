# TrimCI_Flow/dmet/runners/run_dmet_1shot.py
"""
Runner: 1-shot non-overlapping DMET on Fe4S4.

Usage (from /home/unfunnypanda/Proj_Flow/):
    source qflowenv/bin/activate
    python TrimCI_Flow/dmet/runners/run_dmet_1shot.py 2>&1 | tee output.log
"""
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

FCIDUMP = (
    "/home/unfunnypanda/Proj_Flow/"
    "Fe4S4_251230orbital_-327.1920_10kdets/"
    "Fe4S4_251230orbital_-327.1920_10kdets/"
    "fcidump_cycle_6"
)

TRIMCI_CONFIG = {
    "threshold":        0.06,
    "max_final_dets":   "auto",
    "max_rounds":       2,
    "num_runs":         1,
    "pool_build_strategy": "heat_bath",
    "verbose":          False,
}

if __name__ == "__main__":
    from TrimCI_Flow.dmet import run_dmet_1shot

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(
        os.path.dirname(__file__), '..', '..', 'Outputs', 'dmet',
        f"outs_dmet_1shot_{timestamp}")
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"[runner] Output dir: {output_dir}")
    result = run_dmet_1shot(
        fcidump_path  = FCIDUMP,
        trimci_config = TRIMCI_CONFIG,
        output_dir    = output_dir,
    )
    print(f"[runner] Done. E_DMET = {result.E_dmet:.6f} Ha")
