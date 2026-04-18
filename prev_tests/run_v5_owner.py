"""
Run Phase B v5: owner-fragment gamma assembly, N_AVG=5, damping=0.1.

Output: outs_v5_owner_avg5_200dets/
"""
import sys, os, json, datetime
import numpy as np

PROJECT_ROOT = os.path.expanduser('~/Proj_Flow')
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

OUTPUT_DIR = 'outs_v5_owner_avg5_200dets'
os.makedirs(OUTPUT_DIR, exist_ok=True)

from TrimCI_Flow.trimci_flow_v5 import run_selfconsistent_fragments_v5

FCIDUMP = (
    'Fe4S4_251230orbital_-327.1920_10kdets/'
    'Fe4S4_251230orbital_-327.1920_10kdets/'
    'fcidump_cycle_6'
)
TRIMCI_CONFIG = {
    'threshold':           0.03,
    'max_final_dets':      200,
    'max_rounds':          2,
    'num_runs':            1,
    'pool_build_strategy': 'heat_bath',
    'verbose':             False,
}

print(f'Starting v5 owner-fragment run at {datetime.datetime.now().isoformat()}')
result = run_selfconsistent_fragments_v5(
    FCIDUMP,
    window_size=15,
    stride=10,
    max_iterations=60,
    convergence=1e-6,
    rdm_convergence=1e-4,
    damping=0.1,
    n_gamma_avg=5,
    trimci_config=TRIMCI_CONFIG,
    log_bare_reference=True,
)
print(f'Finished at {datetime.datetime.now().isoformat()}')

# --- Iteration table ---
header = f"{'Iter':>5} | {'max|dE| Ha':>12} | {'max|dgamma|':>12} | {'rdm_std10':>12} | {'ndets':>8} | energies"
rows = [header, '-' * len(header)]
for entry in result.iteration_history:
    rows.append(
        f"  {entry['iteration']:3d} | {entry['delta_E']:12.3e} | "
        f"{entry['delta_rdm']:12.3e} | {entry['rdm_rolling_std']:12.3e} | "
        f"{sum(entry['n_dets']):8d} | "
        f"{[f'{e:.3f}' for e in entry['energies']]}"
    )
table_str = '\n'.join(rows)
with open(f'{OUTPUT_DIR}/iteration_table.txt', 'w') as f:
    f.write(table_str + '\n')
print(table_str)

# --- Metadata ---
hist = result.iteration_history
late = [h for h in hist if h['iteration'] >= 40]
very_late = [h for h in hist if h['iteration'] >= 50]

metadata = {
    'run_type':               'Phase_B_v5_owner_avg5',
    'timestamp':              datetime.datetime.now().isoformat(),
    'fcidump':                FCIDUMP,
    'window_size':            15,
    'stride':                 10,
    'max_iterations':         60,
    'convergence':            1e-6,
    'rdm_convergence':        1e-4,
    'damping':                0.1,
    'n_gamma_avg':            5,
    'gamma_assembly':         'owner_fragment',
    'trimci_config':          TRIMCI_CONFIG,
    'converged':              result.converged,
    'iterations_performed':   result.iterations,
    'total_dets_final':       result.total_dets,
    'fragment_n_dets_final':  result.fragment_n_dets,
    'fragment_energies_final': result.fragment_energies,
    'convergence_delta':      result.convergence_delta,
    'convergence_delta_rdm':  result.convergence_delta_rdm,
    'bare_reference':         result._bare_reference,
}

if late:
    drdm = [h['delta_rdm'] for h in late]
    de   = [h['delta_E']   for h in late]
    metadata['late_stats_40_60'] = {
        'delta_rdm_min':    float(np.min(drdm)),
        'delta_rdm_median': float(np.median(drdm)),
        'delta_rdm_max':    float(np.max(drdm)),
        'delta_E_min':      float(np.min(de)),
        'delta_E_median':   float(np.median(de)),
        'delta_E_max':      float(np.max(de)),
    }
if very_late:
    drdm_vl = [h['delta_rdm'] for h in very_late]
    de_vl   = [h['delta_E']   for h in very_late]
    metadata['late_stats_50_60'] = {
        'delta_rdm_min':    float(np.min(drdm_vl)),
        'delta_rdm_median': float(np.median(drdm_vl)),
        'delta_rdm_max':    float(np.max(drdm_vl)),
        'delta_E_min':      float(np.min(de_vl)),
        'delta_E_median':   float(np.median(de_vl)),
        'delta_E_max':      float(np.max(de_vl)),
    }

for thr in [0.05, 0.02, 0.01, 0.005]:
    cnt   = sum(1 for h in hist if h['delta_rdm'] < thr)
    first = next((h['iteration'] for h in hist if h['delta_rdm'] < thr), None)
    metadata[f'threshold_dgamma_{thr}'] = {'count': cnt, 'first_iter': first}

with open(f'{OUTPUT_DIR}/run_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
with open(f'{OUTPUT_DIR}/iteration_history.json', 'w') as f:
    json.dump(result.iteration_history, f, indent=2)

print(f'\nSaved to {OUTPUT_DIR}/')
print(f'  converged={result.converged}')
print(f'  iterations={result.iterations}')
print(f'  final max|dE|={result.convergence_delta:.4e} Ha')
print(f'  final max|dgamma|={result.convergence_delta_rdm:.4e}')
print(f'  fragment_n_dets={result.fragment_n_dets}')
