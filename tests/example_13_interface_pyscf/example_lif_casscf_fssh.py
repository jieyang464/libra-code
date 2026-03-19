#!/usr/bin/env python3
"""
LiF SA-CASSCF + Libra FSSH smoke test.

Requested setup:
- molecule: LiF
- basis: sto-3g
- number of states: 5
- SA-CASSCF weights: always equal (1/nstates)
- active space: always full (all electrons, all orbitals)
- dynamics: Tully FSSH, start from S1, zero initial velocity

Run:
    python example_lif_casscf_fssh.py
"""

import os
import sys
import traceback

import numpy as np

if sys.platform == "cygwin":
    from cyglibra_core import Cpp2Py, CMATRIX, CMATRIXList, Random
elif sys.platform == "linux" or sys.platform == "linux2":
    from liblibra_core import Cpp2Py, CMATRIX, CMATRIXList, Random
else:
    raise RuntimeError(f"Unsupported platform: {sys.platform}")

from libra_py import units
import libra_py.dynamics.tsh.compute as tsh_dynamics

# so imports work when this file is launched from tests/example_13_interface_pyscf/
sys.path.insert(0, os.path.dirname(__file__))

# Ensure we load the PySCF tree that has compiled shared libraries.
_THIS_DIR = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
_PYSCF_LOCAL = os.path.join(_REPO_ROOT, "pyscf")
if os.path.isdir(os.path.join(_PYSCF_LOCAL, "pyscf", "lib")):
    if _PYSCF_LOCAL not in sys.path:
        sys.path.insert(0, _PYSCF_LOCAL)

from interface_pyscf.abc import ESParams, MolecularGeometry
from interface_pyscf.pyscf_esbackend_sketch import PySCFSACASSCFBackend


class _ModelOutput:
    pass


def _lif_geometry_from_bond_length(r_bohr: float) -> MolecularGeometry:
    """
    Build a diatomic LiF geometry centered at the origin.

    Li at z = -R/2, F at z = +R/2 (Bohr).
    """
    coords_bohr = np.array(
        [
            [0.0, 0.0, -0.5 * r_bohr],  # Li
            [0.0, 0.0, +0.5 * r_bohr],  # F
        ],
        dtype=float,
    )
    return MolecularGeometry(atom_labels=["Li", "F"], coords_bohr=coords_bohr)


def _bond_projection(atom_tensor: np.ndarray) -> float:
    """
    Convert atomic z-components to derivative along bond coordinate R.

    For Li(-R/2), F(+R/2):
        d/dR = 0.5 * (d/dz_F - d/dz_L)
    """
    return 0.5 * (float(atom_tensor[1, 2]) - float(atom_tensor[0, 2]))


def compute_lif_model(q, params, full_id):
    """
    Libra compute_model callback backed by stateful PySCF SA-CASSCF.
    """
    idx = Cpp2Py(full_id)[-1]
    r_bohr = float(q.col(idx).get(0))

    try:
        backend: PySCFSACASSCFBackend = params["backend"]
        nstates = int(params["nstates"])
        energy_only_mode = bool(params.get("energy_only_mode", True))

        backend.set_geom(_lif_geometry_from_bond_length(r_bohr))

        if energy_only_mode:
            # Fast smoke-test mode: build SA-CASSCF energies only.
            backend._ensure_core()
            energies = np.asarray(backend._cache.energies[:nstates], dtype=float).copy()
            d1_adi = np.zeros(nstates, dtype=float)
        else:
            energies = np.zeros(nstates, dtype=float)
            d1_adi = np.zeros(nstates, dtype=float)
            for istate in range(nstates):
                eg = backend.energy_gradient(state_id=istate)
                energies[istate] = float(eg.energies[0])
                if eg.gradients is None:
                    raise RuntimeError("State gradients are required for FSSH force evaluation")
                d1_adi[istate] = _bond_projection(np.asarray(eg.gradients[0], dtype=float))

        nac_scalar = np.zeros((nstates, nstates), dtype=float)
        if params.get("use_nacme", False):
            nacres = backend.analytical_nacme(state_id=0)
            if nacres is not None:
                nac = np.asarray(nacres.nacme, dtype=float)
                if nac.shape[:2] != (nstates, nstates):
                    raise RuntimeError(f"Unexpected NAC shape: {nac.shape}")
                for i in range(nstates):
                    for j in range(nstates):
                        nac_scalar[i, j] = _bond_projection(nac[i, j])
    except Exception:
        traceback.print_exc()
        raise

    obj = _ModelOutput()
    obj.ham_adi = CMATRIX(nstates, nstates)
    obj.hvib_adi = CMATRIX(nstates, nstates)
    obj.d1ham_adi = CMATRIXList()
    obj.dc1_adi = CMATRIXList()
    obj.basis_transform = CMATRIX(nstates, nstates)
    obj.time_overlap_adi = CMATRIX(nstates, nstates)
    obj.d1ham_adi.append(CMATRIX(nstates, nstates))
    obj.dc1_adi.append(CMATRIX(nstates, nstates))
    obj.basis_transform.identity()
    obj.time_overlap_adi.identity()

    for i in range(nstates):
        obj.ham_adi.set(i, i, energies[i] * (1.0 + 0.0j))
        obj.hvib_adi.set(i, i, energies[i] * (1.0 + 0.0j))
        obj.d1ham_adi[0].set(i, i, d1_adi[i] * (1.0 + 0.0j))
        for j in range(nstates):
            obj.dc1_adi[0].set(i, j, nac_scalar[i, j] * (1.0 + 0.0j))

    return obj


def main() -> None:
    nstates = 5
    initial_state = 1  # S1
    r0_bohr = 3.0

    # ===================== PySCF settings start =====================
    # Electronic-structure controls live in ESParams + backend object.
    params = ESParams(
        nstates=nstates,
        basis="sto-3g",
        charge=0,
        spin_multiplicity=1,  # singlet only
    )
    backend = PySCFSACASSCFBackend(params=params)
    # ====================== PySCF settings end ======================

    # ===================== Libra settings start =====================
    # Model adapter handle passed to Libra dynamics.
    model_params = {
        "model0": 1,      # required by generic_recipe
        "nstates": nstates,
        "backend": backend,
        "energy_only_mode": True,
        "use_nacme": False,  # set True to request analytical NACs from PySCF
    }

    # Reduced mass for Li-F bond coordinate
    m_li_amu = 6.941
    m_f_amu = 18.998403163
    mu_amu = (m_li_amu * m_f_amu) / (m_li_amu + m_f_amu)

    init_nucl = {
        "ndof": 1,
        "init_type": 0,  # exact (no sampling)
        "q": [r0_bohr],
        "p": [0.0],      # zero velocity
        "mass": [mu_amu * units.amu],
        "force_constant": [0.01],
    }

    istates = [0.0] * nstates
    istates[initial_state] = 1.0
    init_elec = {
        "ndia": nstates,
        "nadi": nstates,
        "init_type": 3,
        "rep": 1,  # adiabatic amplitudes
        "istate": initial_state,
        "istates": istates,
        "init_dm_type": 0,
        "verbosity": 0,
    }

    dyn_params = {
        "nsteps": 1,  # smoke test length
        "dt": 10.0,
        "ntraj": 1,
        "ham_update_method": 2,   # update adiabatic Hamiltonian directly
        "ham_transform_method": 0,
        "rep_tdse": 1,
        "rep_sh": 1,
        "rep_force": 1,
        "force_method": 0,
        "time_overlap_method": 0,
        "nac_update_method": 0,
        "hvib_update_method": 0,
        "state_tracking_algo": 0,
        "do_phase_correction": 0,
        "tsh_method": 0,          # FSSH
        "hop_acceptance_algo": 10,
        "momenta_rescaling_algo": 100,
        "use_boltz_factor": 0,
        "isNBRA": 0,
        "is_nbra": 0,
        "prefix": "lif_casscf_fssh",
        "prefix2": "lif_casscf_fssh",
        "progress_frequency": 1.0,
        "mem_output_level": 1,
        "hdf5_output_level": -1,
        "txt_output_level": 1,
        "txt2_output_level": -1,
        "properties_to_save": [
            "timestep",
            "time",
            "Epot_ave",
            "Ekin_ave",
            "Etot_ave",
        ],
        "which_adi_states": list(range(nstates)),
        "which_dia_states": list(range(nstates)),
    }

    rnd = Random()
    tsh_dynamics.generic_recipe(
        dyn_params, compute_lif_model, model_params, init_elec, init_nucl, rnd
    )
    # ====================== Libra settings end ======================
    print("LiF SA-CASSCF/FSSH test run completed.")
    print("Output directory: lif_casscf_fssh")


if __name__ == "__main__":
    main()
