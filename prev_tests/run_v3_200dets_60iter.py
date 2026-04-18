import sys, os, json, datetime, math
import numpy as np
sys.path.insert(0, '/home/unfunnypanda/Proj_Flow')
os.chdir('/home/unfunnypanda/Proj_Flow')

from TrimCI_Flow.trimci_flow_v3 import run_selfconsistent_fragments_v3

FCIDUMP = ('Fe4S4_251230orbital_-327.1920_10kdets/'
           'Fe4S4_251230orbital_-327.1920_10kdets/fcidump_cycle_6')
OUTPUT_DIR = 'outs_v3_embedded_200dets_60iter'
os.makedirs(OUTPUT_DIR, exist_ok=True)

config_200 = {
    'threshold': 0.03, 'max_final_dets': 200, 'max_rounds': 2,
    'num_runs': 1, 'pool_build_strategy': 'heat_bath', 'verbose': False,
}

print(f'Starting Exp A at {datetime.datetime.now().isoformat()}')
result = run_selfconsistent_fragments_v3(
    FCIDUMP, window_size=15, stride=10,
    max_iterations=60, convergence=1e-6, rdm_convergence=1e-4,
    damping=0.1, n_gamma_avg=1,
    trimci_config=config_200, log_bare_reference=True,
)
print(f'Finished at {datetime.datetime.now().isoformat()}')
print(f'converged={result.converged}, iterations={result.iterations}')

# --- run_metadata.json ---
metadata = {
    'run_type': 'Phase_B_v3_embedded_200dets_60iter',
    'v3_changes': ['embedded_iter0', 'rolling_std_logging', 'out_of_band_bare_ref'],
    'n_gamma_avg': 1,
    'timestamp': datetime.datetime.now().isoformat(),
    'fcidump': FCIDUMP,
    'window_size': 15, 'stride': 10,
    'max_iterations': 60, 'convergence': 1e-6, 'rdm_convergence': 1e-4,
    'damping': 0.1,
    'trimci_config': config_200,
    'converged': bool(result.converged),
    'iterations_performed': int(result.iterations),
    'total_dets_final': int(result.total_dets),
    'fragment_n_dets_final': [int(n) for n in result.fragment_n_dets],
    'fragment_energies_final': [float(e) for e in result.fragment_energies],
    'convergence_delta': float(result.convergence_delta),
    'convergence_delta_rdm': float(result.convergence_delta_rdm),
    'bare_reference': result._bare_reference,
}
with open(os.path.join(OUTPUT_DIR, 'run_metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=2)
print(f'Saved run_metadata.json')

# --- iteration_history.json ---
def make_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, float):
        if math.isinf(obj) or math.isnan(obj):
            return None
        return obj
    elif isinstance(obj, np.floating):
        v = float(obj)
        if math.isinf(v) or math.isnan(v):
            return None
        return v
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return make_serializable(obj.tolist())
    return obj

history_serializable = make_serializable(result.iteration_history)
with open(os.path.join(OUTPUT_DIR, 'iteration_history.json'), 'w') as f:
    json.dump(history_serializable, f, indent=2)
print(f'Saved iteration_history.json ({len(result.iteration_history)} iterations)')

# --- iteration_table.txt ---
lines = []
lines.append(f"{'iter':>4}  {'delta_E':>12}  {'delta_rdm':>12}  {'rdm_std10':>12}  {'total_dets':>10}  energies")
lines.append("-" * 120)
for h in result.iteration_history:
    it        = h['iteration']
    dE        = h['delta_E']
    drdm      = h['delta_rdm']
    std10     = h['rdm_rolling_std']
    tdets     = sum(h['n_dets'])
    eng_str   = '  '.join(f'{e:.6f}' for e in h['energies'])
    dE_s      = 'inf' if (dE is None or (isinstance(dE, float) and math.isinf(dE))) else f'{dE:.4e}'
    drdm_s    = 'inf' if (drdm is None or (isinstance(drdm, float) and math.isinf(drdm))) else f'{drdm:.4e}'
    std10_s   = 'nan' if (std10 is None or (isinstance(std10, float) and math.isnan(std10))) else f'{std10:.4e}'
    lines.append(f"{it:>4}  {dE_s:>12}  {drdm_s:>12}  {std10_s:>12}  {tdets:>10}  {eng_str}")
with open(os.path.join(OUTPUT_DIR, 'iteration_table.txt'), 'w') as f:
    f.write('\n'.join(lines) + '\n')
print(f'Saved iteration_table.txt')

# --- Print final summary ---
print()
print("=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

# Bare reference
if result._bare_reference:
    br = result._bare_reference
    print(f"\nBare reference energies: {[f'{e:.6f}' for e in br['energies']]}")
    print(f"Bare reference n_dets:   {br['n_dets']}")
else:
    print("\nNo bare reference logged.")

# Iter 0
if len(result.iteration_history) > 0:
    h0 = result.iteration_history[0]
    print(f"\nIter 0 (first embedded) energies: {[f'{e:.6f}' for e in h0['energies']]}")
    print(f"  n_dets: {h0['n_dets']}")

# Iter 1
if len(result.iteration_history) > 1:
    h1 = result.iteration_history[1]
    max_dE_1 = h1['delta_E']
    print(f"\nIter 1 energies: {[f'{e:.6f}' for e in h1['energies']]}")
    print(f"  max|ΔE|: {max_dE_1:.4e}")

# Final iter
hlast = result.iteration_history[-1]
final_dE    = result.convergence_delta
final_drdm  = result.convergence_delta_rdm
final_std10 = hlast['rdm_rolling_std']
print(f"\nFinal iter ({hlast['iteration']}) energies: {[f'{e:.6f}' for e in hlast['energies']]}")
print(f"  max|ΔE|:   {final_dE:.4e}")
print(f"  max|Δγ|:   {final_drdm:.4e}")
std10_str = f'{final_std10:.4e}' if (final_std10 is not None and not (isinstance(final_std10, float) and math.isnan(final_std10))) else 'nan'
print(f"  rdm_std10: {std10_str}")

print(f"\nConverged: {result.converged}")
print(f"Iterations performed: {result.iterations}")

# Comparison to previous run
print()
print("Comparison to previous run (outs_phaseB_refinit_damp01_200dets):")
print(f"  Previous:  max|ΔE|=0.233, max|Δγ|=0.083")
print(f"  This run:  max|ΔE|={final_dE:.3f}, max|Δγ|={final_drdm:.3f}")
if final_dE < 0.233:
    print(f"  -> Energy convergence IMPROVED ({final_dE:.3f} < 0.233)")
else:
    print(f"  -> Energy convergence did NOT improve ({final_dE:.3f} >= 0.233)")
if final_drdm < 0.083:
    print(f"  -> RDM convergence IMPROVED ({final_drdm:.3f} < 0.083)")
else:
    print(f"  -> RDM convergence did NOT improve ({final_drdm:.3f} >= 0.083)")

print()
print(f"Output files saved to: {OUTPUT_DIR}/")
print(f"  run_metadata.json")
print(f"  iteration_history.json")
print(f"  iteration_table.txt")
