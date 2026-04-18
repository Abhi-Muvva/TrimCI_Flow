"""Experiment 2: Same as Exp 1 but with max_final_dets=200 and threshold=0.03."""
import json
import os
from datetime import datetime

from TrimCI_Flow.trimci_flow_v2 import run_selfconsistent_fragments_v2

OUT_DIR = "outs_phaseB_refinit_damp01_200dets"
FCIDUMP = "Fe4S4_251230orbital_-327.1920_10kdets/Fe4S4_251230orbital_-327.1920_10kdets/fcidump_cycle_6"

os.makedirs(OUT_DIR, exist_ok=True)

trimci_config_200 = {
    "threshold": 0.03,
    "max_final_dets": 200,
    "max_rounds": 2,
    "num_runs": 1,
    "pool_build_strategy": "heat_bath",
    "verbose": False,
}

timestamp = datetime.now().isoformat()

result = run_selfconsistent_fragments_v2(
    FCIDUMP,
    window_size=15, stride=10,
    max_iterations=40, convergence=1e-6,
    rdm_convergence=1e-4, damping=0.1,
    trimci_config=trimci_config_200,
)

meta = {
    "run_type": "Phase_B_refinit_damp01_200dets",
    "formula": "J - 0.5K (spin-summed gamma)",
    "changes_over_v1": [
        "gamma_mixed initialized from reference determinant (not None)",
        "Global gamma renormalized to conserve electron count",
        "Damping unconditionally applied each iteration",
    ],
    "timestamp": timestamp,
    "fcidump": FCIDUMP,
    "window_size": 15,
    "stride": 10,
    "max_iterations": 40,
    "convergence": 1e-6,
    "rdm_convergence": 1e-4,
    "damping": 0.1,
    "trimci_config": trimci_config_200,
    "converged": bool(result.converged),
    "iterations_performed": int(result.iterations),
    "total_dets_final": int(result.total_dets),
    "fragment_n_dets_final": [int(x) for x in result.fragment_n_dets],
    "fragment_energies_final": [float(x) for x in result.fragment_energies],
    "convergence_delta": float(result.convergence_delta),
    "convergence_delta_rdm": float(result.convergence_delta_rdm),
}
with open(os.path.join(OUT_DIR, "run_metadata.json"), "w") as f:
    json.dump(meta, f, indent=2)

hist_out = []
for entry in result.iteration_history:
    e = dict(entry)
    if e.get("delta_E") is None or (isinstance(e.get("delta_E"), float) and (e["delta_E"] != e["delta_E"] or e["delta_E"] == float('inf'))):
        e["delta_E"] = None
    if e.get("delta_rdm") is None or (isinstance(e.get("delta_rdm"), float) and (e["delta_rdm"] != e["delta_rdm"] or e["delta_rdm"] == float('inf'))):
        e["delta_rdm"] = None
    e["energies"] = [float(x) for x in e["energies"]]
    e["n_dets"] = [int(x) for x in e["n_dets"]]
    hist_out.append(e)
with open(os.path.join(OUT_DIR, "iteration_history.json"), "w") as f:
    json.dump(hist_out, f, indent=2)

lines = []
lines.append("iter | delta_E (Ha)   | delta_rdm     | total_dets | fragment_energies")
lines.append("-" * 90)
for entry in result.iteration_history:
    it = entry["iteration"]
    dE = entry["delta_E"]
    dR = entry["delta_rdm"]
    total_dets = sum(entry["n_dets"])
    energies = entry["energies"]
    dE_s = "         inf" if (dE is None or dE == float('inf')) else f"{dE:12.3e}"
    dR_s = "         inf" if (dR is None or dR == float('inf')) else f"{dR:12.3e}"
    e_s = "[" + ", ".join(f"{x:.3f}" for x in energies) + "]"
    lines.append(f"  {it:2d}  | {dE_s} | {dR_s} | {total_dets:8d} | {e_s}")
with open(os.path.join(OUT_DIR, "iteration_table.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"\nExperiment 2 complete. converged={result.converged}, iters={result.iterations}")
print(f"Final delta_E={result.convergence_delta:.3e}, delta_rdm={result.convergence_delta_rdm:.3e}")
print(f"Saved to: {OUT_DIR}/")
