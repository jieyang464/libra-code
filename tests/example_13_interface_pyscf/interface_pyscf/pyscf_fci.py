# *********************************************************************************
# * Copyright (C) 2026 Jie Yang, Alexey V. Akimov
# *
# * This file is distributed under the terms of the GNU General Public License
# * as published by the Free Software Foundation, either version 3 of
# * the License, or (at your option) any later version.
# * See the file LICENSE in the root directory of this distribution
# * or <http://www.gnu.org/licenses/>.
# *
# *********************************************************************************/
"""
pyscf_fci – Concrete PySCF backend: RHF -> Full CI.

Implements the current ``ESBackend`` ABI in ``abc.py``.
One backend instance holds parameters and mutable geometry via ``set_geom``.

Usage::

    from interface_pyscf.abc import MolecularGeometry, ESParams
    from interface_pyscf.pyscf_fci import PySCFFCIBackend

    geom = MolecularGeometry(
        atom_labels=["H", "H"],
        coords_bohr=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]],
    )
    params = ESParams(nstates=2, basis="sto-3g")

    backend = PySCFFCIBackend(params=params, geometry=geom)
    eg = backend.energy_gradient(state_id=0)
    wfn = backend.wavefunction(state_id=0)
    nac = backend.analytical_nacme(state_id=0)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from pyscf import fci, gto, scf

from .abc import (
    ESBackend,
    ESParams,
    EnergyGradientResult,
    MolecularGeometry,
    NACMEResult,
    WavefunctionResult,
)

_BOHR_TO_ANG = 0.529177210903


def _build_mol(geometry: MolecularGeometry, params: ESParams) -> gto.Mole:
    atom_list = [
        [sym, tuple(coord * _BOHR_TO_ANG)]
        for sym, coord in zip(geometry.atom_labels, geometry.coords_bohr)
    ]
    return gto.M(
        atom=atom_list,
        basis=params.basis,
        charge=params.charge,
        spin=params.spin_multiplicity - 1,  # PySCF spin = 2S
        unit="Angstrom",
        verbose=0,
    )


@dataclass
class _FCICache:
    core_done: bool = False
    gradients_done: bool = False
    nacme_done: bool = False

    energies: Optional[np.ndarray] = None
    gradients: Optional[np.ndarray] = None
    nacme: Optional[np.ndarray] = None

    mo_coefficients: Optional[np.ndarray] = None
    ao_overlap: Optional[np.ndarray] = None
    ci_coefficients: Optional[np.ndarray] = None

    mf: object = None
    ci_solver: object = None


class PySCFFCIBackend(ESBackend):
    """
    Lazy cached RHF->FCI backend with mutable geometry.

    Shared lazy core avoids recomputing RHF/FCI across:
    - ``energy_gradient(state_id)``
    - ``wavefunction(state_id)``
    - ``analytical_nacme(state_id)``
    """

    def __init__(
        self,
        params: ESParams,
        geometry: Optional[MolecularGeometry] = None,
        gradient_step: float = 1e-4,
    ) -> None:
        self._geometry = geometry
        self._params = params
        self._cache = _FCICache()
        self._gradient_step = gradient_step

    def _reset_cache(self) -> None:
        self._cache = _FCICache()

    def set_geom(self, geometry: MolecularGeometry) -> None:
        self._geometry = geometry
        self._reset_cache()

    def _validate_state(self, state_id: int) -> int:
        if state_id < 0 or state_id >= self._params.nstates:
            raise ValueError(
                f"state_id={state_id} out of range [0, {self._params.nstates - 1}]"
            )
        return state_id

    def _ensure_core(self) -> None:
        if self._cache.core_done:
            return
        if self._geometry is None:
            raise RuntimeError("Geometry is not set. Call set_geom(...) before querying.")

        mol = _build_mol(self._geometry, self._params)

        mf = scf.RHF(mol)
        mf.verbose = 0
        mf.kernel()
        if not mf.converged:
            raise RuntimeError("RHF did not converge")

        ci_solver = fci.FCI(mf)
        ci_solver.nroots = self._params.nstates
        ci_solver.verbose = 0
        e_ci, ci_vecs = ci_solver.kernel()

        if self._params.nstates == 1:
            energies = np.atleast_1d(e_ci)
            ci_vecs = [ci_vecs]
        else:
            energies = np.asarray(e_ci)

        ci_flat = [np.asarray(v).ravel() for v in ci_vecs]
        ci_matrix = np.column_stack(ci_flat)

        self._cache.energies = energies
        self._cache.mo_coefficients = np.asarray(mf.mo_coeff)
        self._cache.ao_overlap = np.asarray(mol.intor("int1e_ovlp"))
        self._cache.ci_coefficients = ci_matrix
        self._cache.mf = mf
        self._cache.ci_solver = ci_solver
        self._cache.core_done = True

    def _fci_energies_for_geometry(self, geometry: MolecularGeometry) -> np.ndarray:
        mol = _build_mol(geometry, self._params)
        mf = scf.RHF(mol)
        mf.verbose = 0
        mf.kernel()
        if not mf.converged:
            raise RuntimeError("RHF did not converge for displaced geometry")

        ci_solver = fci.FCI(mf)
        ci_solver.nroots = self._params.nstates
        ci_solver.verbose = 0
        e_ci, _ = ci_solver.kernel()
        if self._params.nstates == 1:
            return np.atleast_1d(e_ci)
        return np.asarray(e_ci)

    def _numerical_gradient(self, istate: int) -> np.ndarray:
        natoms = len(self._geometry.atom_labels)
        grad = np.empty((natoms, 3))
        coords = self._geometry.coords_bohr.copy()
        step = self._gradient_step

        for atom in range(natoms):
            for axis in range(3):
                coords_p = coords.copy()
                coords_m = coords.copy()
                coords_p[atom, axis] += step
                coords_m[atom, axis] -= step

                geom_p = MolecularGeometry(self._geometry.atom_labels, coords_p)
                geom_m = MolecularGeometry(self._geometry.atom_labels, coords_m)

                e_p = self._fci_energies_for_geometry(geom_p)
                e_m = self._fci_energies_for_geometry(geom_m)
                grad[atom, axis] = (e_p[istate] - e_m[istate]) / (2.0 * step)

        return grad

    def _ensure_gradients(self) -> None:
        if self._cache.gradients_done:
            return
        self._ensure_core()

        natoms = len(self._geometry.atom_labels)
        gradients = np.empty((self._params.nstates, natoms, 3))
        for istate in range(self._params.nstates):
            try:
                grad_obj = self._cache.ci_solver.nuc_grad_method()
                grad_obj.verbose = 0
                gradients[istate] = grad_obj.kernel(state=istate)
            except Exception:
                gradients[istate] = self._numerical_gradient(istate)
        self._cache.gradients = gradients
        self._cache.gradients_done = True

    def _ensure_nacme(self) -> None:
        if self._cache.nacme_done:
            return
        self._ensure_core()

        # RHF->FCI NACME is left as backend-dependent extension.
        self._cache.nacme = None
        self._cache.nacme_done = True

    def energy_gradient(self, state_id: int) -> EnergyGradientResult:
        sid = self._validate_state(state_id)
        self._ensure_core()
        self._ensure_gradients()
        return EnergyGradientResult(
            energies=self._cache.energies[sid:sid+1].copy(),
            gradients=self._cache.gradients[sid:sid+1].copy(),
        )

    def wavefunction(self, state_id: int) -> WavefunctionResult:
        sid = self._validate_state(state_id)
        self._ensure_core()
        return WavefunctionResult(
            mo_coefficients=self._cache.mo_coefficients.copy(),
            ao_overlap=self._cache.ao_overlap.copy(),
            ci_coefficients=self._cache.ci_coefficients[:, sid:sid+1].copy(),
            overlap=None,
        )

    def analytical_nacme(self, state_id: int) -> Optional[NACMEResult]:
        sid = self._validate_state(state_id)
        self._ensure_core()
        self._ensure_nacme()
        if self._cache.nacme is None:
            return None
        return NACMEResult(nacme=self._cache.nacme[sid:sid+1].copy())
