# DMET SC — 2026-04-16 01:23

**Status:** NOT_CONVERGED  |  **SC iterations:** 5  |  **Elapsed:** 6.6 min

## Parameters

- `max_sc_iter` = 5
- `conv_tol`    = 0.001
- `u_damp`      = 0.5
- `u_step`      = -0.25
- TrimCI `threshold`      = 0.06
- TrimCI `max_final_dets` = auto
- TrimCI `max_rounds`     = 2

## Final energies

| Quantity | Value (Ha) |
|----------|-----------|
| E_HF (UHF, last iter) | -321.636473 |
| E_DMET_A (1-body, debug) | -584.998545 |
| **E_DMET_B (2-RDM, primary)** | **-143.578159** |
| E_DMET_C (democratic) | -122.260661 |
| Error vs −327.1920 | +183.6138 |

Total dets: 150  (fragments: [50, 50, 50])

## SC convergence history

| iter | E_DMET_B (Ha) | max\|Δγ_frag\| | max\|Δu\| | total dets | conv |
|-----:|-------------:|---------------:|----------:|----------:|------|
|   0 |     -261.791719 | 1.3140e+00 | 0.0000e+00 |   150 |  |
|   1 |     -145.500878 | 1.8820e+00 | 1.6425e-01 |   150 |  |
|   2 |     -202.420835 | 1.9876e+00 | 2.3524e-01 |   150 |  |
|   3 |     -217.843576 | 1.9772e+00 | 2.4845e-01 |   150 |  |
|   4 |     -143.578159 | 1.9804e+00 | 2.4714e-01 |   150 |  |

## Fragment diagnostics (final iteration)

- **F0**: n_imp=21 (frag=12, bath=9, core=15), n_elec_imp=24, E_imp=-81.507657 Ha, E_B=-49.657620 Ha, RDM disc=5.68e-14 Ha
- **F1**: n_imp=21 (frag=12, bath=9, core=15), n_elec_imp=24, E_imp=-71.205935 Ha, E_B=-48.572341 Ha, RDM disc=4.26e-14 Ha
- **F2**: n_imp=21 (frag=12, bath=9, core=15), n_elec_imp=24, E_imp=-61.242564 Ha, E_B=-45.348199 Ha, RDM disc=1.28e-13 Ha

---
