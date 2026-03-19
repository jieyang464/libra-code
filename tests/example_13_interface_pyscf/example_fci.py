#!/usr/bin/env python3
"""
example_fci.py – Minimal demo of the PySCF FCI backend.

Run::

    python example_fci.py

Expected output:  FCI energies and MO/CI shapes for H₂ / sto-3g.
"""
import sys, os
import numpy as np

# so the import works from inside tests/example_13_interface_pyscf/
sys.path.insert(0, os.path.dirname(__file__))

from interface_pyscf.abc import MolecularGeometry, ESParams
from interface_pyscf.pyscf_fci import PySCFFCIBackend


def main():
    # ── build geometry + params ───────────────────────────────────
    geom = MolecularGeometry(
        atom_labels=["H", "H"],
        coords_bohr=np.array([[0.0, 0.0, 0.0],
                               [0.0, 0.0, 1.4]]),  # ~0.74 Å
    )

    params = ESParams(
        nstates=2,
        basis="sto-3g",
    )

    # ── instantiate backend (bound to one geometry) ───────────────
    backend = PySCFFCIBackend(params=params)
    backend.set_geom(geom)

    # ── concern 1: energies + gradients ────────────────────────────
    eg = backend.energy_gradient(state_id=0)
    print("FCI energies (hartree):")
    for i, e in enumerate(eg.energies):
        print(f"  state {i}: {e:16.10f}")
    print()

    # ── concern 2: wavefunction data ───────────────────────────────
    wfn = backend.wavefunction(state_id=0)
    print(f"MO coefficients shape : {wfn.mo_coefficients.shape}")
    print(f"AO overlap shape      : {wfn.ao_overlap.shape}")
    print(f"CI coefficients shape : {wfn.ci_coefficients.shape}")
    print()

    # Quick sanity: CI vectors are orthonormal
    ovlp = wfn.ci_coefficients.T @ wfn.ci_coefficients
    print("CI vector overlap (should be identity):")
    print(np.array2string(ovlp, precision=6, suppress_small=True))

if __name__ == "__main__":
    main()
