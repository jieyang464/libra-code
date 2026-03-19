from pathlib import Path
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from libra_py.packages.abc import MolecularGeometry
    from libra_py.packages.pyscf.casscf import CASSCFBackend
else:
    from ..abc import MolecularGeometry
    from .casscf import CASSCFBackend


def main() -> None:
    # two geoms for testing for HeH+
    geom1 = MolecularGeometry(
        atom_labels=["He", "H"],
        coords_bohr=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4632]]),
    )
    geom2 = MolecularGeometry(
        atom_labels=["He", "H"],
        coords_bohr=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]]),
    )

    # instantiate the CASSCF backend with 2e 2mo sto-3g
    casscf_backend = CASSCFBackend(norbcas=2, nelecas=2, nroots=3, basis="sto-3g", charge=1)

    # call set_geom_and_run_hf for the first geometry
    casscf_backend.set_geom_and_run_hf(geom1)

    # call compute_energy for the 3 roots
    energies_geom1 = [casscf_backend.compute_energy(root) for root in range(3)]
    print("Energies for geom1:\n", energies_geom1)

    # call set_geom_and_run_hf for the second geometry
    casscf_backend.set_geom_and_run_hf(geom2)

    # print cached data from geom1 to verify the wavefunction info is preserved
    print("Cached WFN info from geom1:")
    print("MO coefficients:\n", casscf_backend._wfncache.mo_coeff)
    print("CI coefficients:\n", casscf_backend._wfncache.ci_coeff)
    print("AO overlap matrix:\n", casscf_backend._wfncache.ao_overlap)

    # call compute_energy for the 3 roots
    energies_geom2 = [casscf_backend.compute_energy(root) for root in range(3)]
    print("Energies for geom2:\n", energies_geom2)

    # call compute_gradient for root 0
    gradients_geom2_root0 = casscf_backend.compute_gradient(root=0)
    print("Gradient for geom2 root 0:\n", gradients_geom2_root0)
    # call compute_gradient for root 1
    gradients_geom2_root1 = casscf_backend.compute_gradient(root=1)
    print("Gradient for geom2 root 1:\n", gradients_geom2_root1)
    # call compute_gradient for root 2
    gradients_geom2_root2 = casscf_backend.compute_gradient(root=2)
    print("Gradient for geom2 root 2:\n", gradients_geom2_root2)

    # compute and print the time-overlap matrix for the 3 roots
    time_overlap_matrix = casscf_backend.time_overlap_matrix(nroots=3)
    print("Time-overlap matrix:\n", time_overlap_matrix)


if __name__ == "__main__":
    main()
