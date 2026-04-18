# DMET SC — 2026-04-16 01:16

**Status:** NOT_CONVERGED  |  **SC iterations:** 15  |  **Elapsed:** 4.8 min

## Parameters

- `max_sc_iter` = 15
- `conv_tol`    = 0.001
- `u_damp`      = 0.5
- `u_step`      = 1.0
- TrimCI `threshold`      = 0.06
- TrimCI `max_final_dets` = auto
- TrimCI `max_rounds`     = 2

## Final energies

| Quantity | Value (Ha) |
|----------|-----------|
| E_HF (UHF, last iter) | -433.119708 |
| E_DMET_A (1-body, debug) | -621.161439 |
| **E_DMET_B (2-RDM, primary)** | **-151.863489** |
| E_DMET_C (democratic) | -119.078463 |
| Error vs −327.1920 | +175.3285 |

Total dets: 150  (fragments: [50, 50, 50])

## SC convergence history

| iter | E_DMET_B (Ha) | max\|Δγ_frag\| | max\|Δu\| | total dets | conv |
|-----:|-------------:|---------------:|----------:|----------:|------|
|   0 |     -254.405458 | 1.2936e+00 | 0.0000e+00 |   150 |  |
|   1 |     -228.899689 | 1.9791e+00 | 6.4682e-01 |   150 |  |
|   2 |     -239.266367 | 1.9833e+00 | 9.8956e-01 |   150 |  |
|   3 |     -160.725942 | 1.7455e+00 | 9.9165e-01 |   150 |  |
|   4 |     -153.954661 | 1.3991e+00 | 8.7273e-01 |   150 |  |
|   5 |     -155.901555 | 1.9195e+00 | 6.9953e-01 |   150 |  |
|   6 |     -156.163032 | 1.7804e+00 | 9.5976e-01 |   150 |  |
|   7 |     -155.803699 | 1.8070e+00 | 8.9018e-01 |   150 |  |
|   8 |     -158.681882 | 1.8292e+00 | 9.0350e-01 |   150 |  |
|   9 |     -151.937262 | 1.5522e+00 | 9.1460e-01 |   150 |  |
|  10 |     -158.309938 | 1.2793e+00 | 7.7609e-01 |   150 |  |
|  11 |     -155.110986 | 1.8307e+00 | 6.3965e-01 |   150 |  |
|  12 |     -157.047033 | 1.5828e+00 | 9.1533e-01 |   150 |  |
|  13 |     -150.846089 | 1.8855e+00 | 7.9140e-01 |   150 |  |
|  14 |     -151.863489 | 1.4633e+00 | 9.4274e-01 |   150 |  |

## Fragment diagnostics (final iteration)

- **F0**: n_imp=21 (frag=12, bath=9, core=15), n_elec_imp=24, E_imp=-77.392578 Ha, E_B=-46.405308 Ha, RDM disc=9.95e-14 Ha
- **F1**: n_imp=21 (frag=12, bath=9, core=15), n_elec_imp=24, E_imp=-72.756253 Ha, E_B=-49.196490 Ha, RDM disc=0.00e+00 Ha
- **F2**: n_imp=21 (frag=12, bath=9, core=15), n_elec_imp=24, E_imp=-58.238479 Ha, E_B=-56.261690 Ha, RDM disc=4.26e-14 Ha

---
