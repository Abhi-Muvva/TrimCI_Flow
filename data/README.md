# Fe4S4 Reference Data

This directory contains the input data for TrimCI_Flow. Both files come from a
converged TrimCI calculation on the Fe4S4 iron-sulfur cluster performed in Dr. Otten's
group (cycle 6 of a self-consistent orbital optimization).

---

## System

**Fe4S4** — an iron-sulfur cubane cluster. These clusters appear throughout biology
(ferredoxins, nitrogenase, photosystem I) as electron-transfer cofactors. Fe4S4 is a
canonical strongly-correlated benchmark: its four iron centers are high-spin and the
d-electrons are delocalized across the cluster in a way that single-reference methods
(Hartree-Fock, DFT) cannot describe accurately. You need many Slater determinants.

| Property | Value |
|----------|-------|
| Active space | 36 spatial orbitals |
| Electrons | 27α + 27β = 54 total |
| Point group | Approximate C₂ᵥ |
| Brute-force TrimCI energy | −327.1920 Ha |
| Brute-force TrimCI det count | 10,095 determinants |

---

## Files

### `fcidump_cycle_6` (9.2 MB)

FCIDUMP-format one- and two-electron integrals for the 36-orbital active space. Standard
Molpro/PySCF FCIDUMP layout:

```
&FCI NORB=36, NELEC=54, MS2=0, ...
 <two-electron integrals: (pq|rs) value  p q r s>
 ...
 <one-electron integrals: h[p,q] value  p q>
 <core energy:            E_nuc  0 0 0 0>
```

**Critical:** `E_nuc = 0.0` in this file. The nuclear repulsion energy has already been
absorbed into the one-body integrals during the orbital optimization. Do not add a
separate nuclear repulsion term.

Read with TrimCI:
```python
import trimci
h1, eri, n_elec, n_orb, E_nuc, n_alpha, n_beta, psym = trimci.read_fcidump("data/fcidump_cycle_6")
# h1:    (36, 36) one-body integrals, chemist notation
# eri:   (36, 36, 36, 36) two-electron integrals, chemist notation: (pq|rs)
# E_nuc: 0.0  — do not add
```

ERI convention: chemist notation throughout. `eri[p, q, r, s] = (pq|rs)`.
8-fold symmetry holds: `eri[p,q,r,s] = eri[q,p,r,s] = eri[p,q,s,r] = eri[r,s,p,q]`.

### `dets.npz` (215 KB)

NumPy compressed archive containing the 10,095 determinants from the brute-force TrimCI
run, stored as packed uint64 bitstrings.

```python
import numpy as np
data = np.load("data/dets.npz")
dets = data["dets"]           # shape (10095, 2), dtype uint64
# dets[i, 0] = alpha bitstring of determinant i
# dets[i, 1] = beta  bitstring of determinant i
```

Bit convention: bit `p` of `alpha_bits` is set if spatial orbital `p` is α-occupied.
Orbital indices are 0-based. For `n_orb=36`, bits 0–35 are used; higher bits are zero.

**Row 0 is the correlated reference determinant.** This is the starting point for TrimCI's
heat-bath expansion — it is NOT the Hartree-Fock determinant. Using HF instead of row 0
for fragment electron counting gives wrong occupations near the Fermi level (all
low-energy orbitals filled, all high-energy orbitals empty), collapsing fragments to
trivial one-determinant solutions. Always use `dets[0]`.

```python
ref_alpha = int(dets[0, 0])
ref_beta  = int(dets[0, 1])
# Check occupation of orbital p:
is_alpha_occ = bool((ref_alpha >> p) & 1)
```

---

## Provenance

- Orbitals: CASSCF-optimized, cycle 6 of a self-consistent loop
- TrimCI run: heat-bath selected CI with threshold 0.06, auto det cap
- Reference energy −327.1920 Ha is the converged TrimCI total energy at this geometry
- Data provided by Dr. Otten's group; used here as the benchmark for fragmentation experiments
