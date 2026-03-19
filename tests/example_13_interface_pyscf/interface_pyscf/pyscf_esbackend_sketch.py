"""
pyscf_esbackend_sketch
======================

Reference implementation for the current ``ESBackend`` ABI.

Goal:
- avoid duplicate PySCF work when callers invoke
  ``energy_gradient(state_id)``, ``wavefunction(state_id)``, and
  ``analytical_nacme(state_id)`` in any order for the same geometry.
- show when backend-level state is useful (previous-step wavefunction
  for overlap tracking).
- show how a dataclass-of-callables can reuse the same cache if all
  callables share one underlying backend instance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .abc import (
    ESBackend,
    ESParams,
    EnergyGradientResult,
    MolecularGeometry,
    NACMEResult,
    WavefunctionResult,
)

_BOHR_TO_ANG = 0.529177210903


@dataclass
class _SolveCache:
    """Per-geometry cache owned by one backend object."""

    core_done: bool = False
    gradients_done: bool = False
    nacme_done: bool = False

    energies: Optional[np.ndarray] = None
    gradients: Optional[np.ndarray] = None
    nacme: Optional[np.ndarray] = None

    mo_coefficients: Optional[np.ndarray] = None
    ao_overlap: Optional[np.ndarray] = None
    ci_coefficients: Optional[np.ndarray] = None

    # Keep solver objects only as long as this backend is alive.
    mf: object = None
    ci_solver: object = None


class PySCFESBackend(ESBackend):
    """Lazy, cached PySCF backend with mutable geometry."""

    def __init__(
        self,
        params: ESParams,
        geometry: Optional[MolecularGeometry] = None,
    ) -> None:
        self._geometry = geometry
        self._params = params
        self._cache = _SolveCache()

    def _reset_cache(self) -> None:
        self._cache = _SolveCache()

    def set_geom(self, geometry: MolecularGeometry) -> None:
        self._geometry = geometry
        self._reset_cache()

    def _validate_state(self, state_id: int) -> int:
        if state_id < 0 or state_id >= self._params.nstates:
            raise ValueError(
                f"state_id={state_id} out of range [0, {self._params.nstates - 1}]"
            )
        return state_id

    def _build_mol(self):
        from pyscf import gto
        if self._geometry is None:
            raise RuntimeError("Geometry is not set. Call set_geom(...) before querying.")

        atom_list = [
            [sym, tuple(coord * _BOHR_TO_ANG)]
            for sym, coord in zip(self._geometry.atom_labels, self._geometry.coords_bohr)
        ]
        return gto.M(
            atom=atom_list,
            basis=self._params.basis,
            charge=self._params.charge,
            spin=self._params.spin_multiplicity - 1,  # PySCF spin = 2S
            unit="Angstrom",
            verbose=0,
        )

    def _ensure_core(self) -> None:
        """Compute SCF + CI once; all public methods depend on this."""
        if self._cache.core_done:
            return

        from pyscf import fci, scf

        mol = self._build_mol()

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

    def _ensure_gradients(self) -> None:
        """Try to compute gradients once; cache None if unsupported."""
        if self._cache.gradients_done:
            return
        self._ensure_core()

        natoms = len(self._geometry.atom_labels)
        gradients = np.empty((self._params.nstates, natoms, 3))
        try:
            for istate in range(self._params.nstates):
                grad_obj = self._cache.ci_solver.nuc_grad_method()
                grad_obj.verbose = 0
                gradients[istate] = grad_obj.kernel(state=istate)
            self._cache.gradients = gradients
        except Exception:
            self._cache.gradients = None
        self._cache.gradients_done = True

    def _ensure_nacme(self) -> None:
        """Compute NACME once from cached core data (placeholder sketch)."""
        if self._cache.nacme_done:
            return
        self._ensure_core()

        # Sketch behavior:
        # - This is where one would call a PySCF NACME implementation.
        # - Keep None when not available for the chosen method.
        self._cache.nacme = None
        self._cache.nacme_done = True

    def energy_gradient(self, state_id: int) -> EnergyGradientResult:
        sid = self._validate_state(state_id)
        self._ensure_core()
        self._ensure_gradients()
        gradients = None
        if self._cache.gradients is not None:
            gradients = self._cache.gradients[sid:sid+1].copy()
        return EnergyGradientResult(
            energies=self._cache.energies[sid:sid+1].copy(),
            gradients=gradients,
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


@dataclass(frozen=True)
class SinglePointCallables:
    """
    Dataclass-of-callables variant.

    Works only if the three callables share one cache. The easiest way to
    guarantee that is to bind them to one ``PySCFESBackend`` instance.
    """

    set_geom: Callable[[MolecularGeometry], None]
    energy_gradient: Callable[[int], EnergyGradientResult]
    wavefunction: Callable[[int], WavefunctionResult]
    analytical_nacme: Callable[[int], Optional[NACMEResult]]


def make_single_point_callables(
    params: ESParams,
    *,
    geometry: Optional[MolecularGeometry] = None,
) -> SinglePointCallables:
    """
    Bridge for experiments without ABC inheritance.

    Note: if each callable were built independently, you'd lose cache sharing
    and recompute SCF/CI three times.
    """

    backend = PySCFESBackend(
        params,
        geometry=geometry,
    )
    return SinglePointCallables(
        set_geom=backend.set_geom,
        energy_gradient=backend.energy_gradient,
        wavefunction=backend.wavefunction,
        analytical_nacme=backend.analytical_nacme,
    )


@dataclass
class _SACASSCFCache:
    """Per-geometry cache for SA-CASSCF backend."""

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
    mc: object = None


class PySCFSACASSCFBackend(ESBackend):
    """
    SA-CASSCF backend with geometry cache and state-resolved API.

    Design choices requested by user workflow:
    - Active space is always full (all electrons, all MOs from chosen basis).
    - State-averaging weights are always uniform (1/nstates).
    """

    def __init__(
        self,
        params: ESParams,
        geometry: Optional[MolecularGeometry] = None,
    ) -> None:
        self._geometry = geometry
        self._params = params
        self._cache = _SACASSCFCache()

    def _reset_cache(self) -> None:
        self._cache = _SACASSCFCache()

    def set_geom(self, geometry: MolecularGeometry) -> None:
        self._geometry = geometry
        self._reset_cache()

    def _validate_state(self, state_id: int) -> int:
        if state_id < 0 or state_id >= self._params.nstates:
            raise ValueError(
                f"state_id={state_id} out of range [0, {self._params.nstates - 1}]"
            )
        return state_id

    def _build_mol(self):
        from pyscf import gto

        if self._geometry is None:
            raise RuntimeError("Geometry is not set. Call set_geom(...) before querying.")

        atom_list = [
            [sym, tuple(coord * _BOHR_TO_ANG)]
            for sym, coord in zip(self._geometry.atom_labels, self._geometry.coords_bohr)
        ]
        return gto.M(
            atom=atom_list,
            basis=self._params.basis,
            charge=self._params.charge,
            spin=self._params.spin_multiplicity - 1,  # PySCF spin = 2S
            unit="Angstrom",
            verbose=0,
        )

    @staticmethod
    def _full_active_space(mf, mol) -> tuple[int, int]:
        """Use all orbitals/electrons as active space."""
        ncas = int(np.asarray(mf.mo_coeff).shape[1])
        nelecas = int(mol.nelectron)
        return ncas, nelecas

    @staticmethod
    def _as_ci_matrix(ci_objects, nstates: int) -> np.ndarray:
        if not isinstance(ci_objects, (list, tuple)):
            ci_objects = [ci_objects]
        if len(ci_objects) < nstates:
            raise RuntimeError(
                f"Expected at least {nstates} CI roots, got {len(ci_objects)}"
            )
        ci_flat = [np.asarray(ci_objects[i]).ravel() for i in range(nstates)]
        return np.column_stack(ci_flat)

    def _ensure_core(self) -> None:
        """Compute RHF + SA-CASSCF once for current geometry."""
        if self._cache.core_done:
            return

        from pyscf import mcscf, scf

        mol = self._build_mol()
        mf = scf.RHF(mol)
        mf.verbose = 0
        mf.kernel()
        if not mf.converged:
            raise RuntimeError("RHF did not converge")

        ncas, nelecas = self._full_active_space(mf, mol)
        mc = mcscf.CASSCF(mf, ncas, nelecas)
        mc.fcisolver.nroots = self._params.nstates
        mc.verbose = 0

        weights = np.full(self._params.nstates, 1.0 / float(self._params.nstates))
        mc = mc.state_average_(weights.tolist())
        mc.kernel()
        if not mc.converged:
            raise RuntimeError("SA-CASSCF did not converge")

        e_states = np.asarray(getattr(mc, "e_states", []), dtype=float)
        if e_states.size == 0:
            e_tot = np.asarray(getattr(mc, "e_tot", []), dtype=float)
            e_states = np.atleast_1d(e_tot)
        if e_states.size < self._params.nstates:
            raise RuntimeError(
                f"Requested {self._params.nstates} states, but solver returned {e_states.size}"
            )

        self._cache.energies = e_states[: self._params.nstates].copy()
        self._cache.mo_coefficients = np.asarray(mc.mo_coeff, dtype=float)
        self._cache.ao_overlap = np.asarray(mol.intor("int1e_ovlp"), dtype=float)
        self._cache.ci_coefficients = self._as_ci_matrix(mc.ci, self._params.nstates)
        self._cache.mf = mf
        self._cache.mc = mc
        self._cache.core_done = True

    def _ensure_gradients(self) -> None:
        if self._cache.gradients_done:
            return
        self._ensure_core()

        natoms = len(self._geometry.atom_labels)
        gradients = np.empty((self._params.nstates, natoms, 3))

        try:
            grad_obj = self._cache.mc.nuc_grad_method()
            grad_obj.verbose = 0
            for istate in range(self._params.nstates):
                gradients[istate] = np.asarray(grad_obj.kernel(state=istate), dtype=float)
        except Exception as exc:
            raise RuntimeError("SA-CASSCF analytical gradients are unavailable") from exc

        self._cache.gradients = gradients
        self._cache.gradients_done = True

    def _ensure_nacme(self) -> None:
        if self._cache.nacme_done:
            return
        self._ensure_core()

        natoms = len(self._geometry.atom_labels)
        nstates = self._params.nstates
        nac = np.zeros((nstates, nstates, natoms, 3))

        try:
            nac_solver = self._cache.mc.nac_method()
            nac_solver.verbose = 0
            for i in range(nstates):
                for j in range(i + 1, nstates):
                    dij = np.asarray(nac_solver.kernel(state=(i, j)), dtype=float)
                    nac[i, j] = dij
                    nac[j, i] = -dij
            self._cache.nacme = nac
        except Exception:
            self._cache.nacme = None

        self._cache.nacme_done = True

    def energy_gradient(self, state_id: int) -> EnergyGradientResult:
        sid = self._validate_state(state_id)
        self._ensure_core()
        self._ensure_gradients()
        return EnergyGradientResult(
            energies=self._cache.energies[sid:sid + 1].copy(),
            gradients=self._cache.gradients[sid:sid + 1].copy(),
        )

    def wavefunction(self, state_id: int) -> WavefunctionResult:
        sid = self._validate_state(state_id)
        self._ensure_core()
        return WavefunctionResult(
            mo_coefficients=self._cache.mo_coefficients.copy(),
            ao_overlap=self._cache.ao_overlap.copy(),
            ci_coefficients=self._cache.ci_coefficients[:, sid:sid + 1].copy(),
            overlap=None,
        )

    def analytical_nacme(self, state_id: int) -> Optional[NACMEResult]:
        self._validate_state(state_id)
        self._ensure_core()
        self._ensure_nacme()
        if self._cache.nacme is None:
            return None
        return NACMEResult(nacme=self._cache.nacme.copy())
