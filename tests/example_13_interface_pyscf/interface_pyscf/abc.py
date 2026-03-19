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
abc – Abstract base classes and data containers for ES backends.

Design
------
One stateful backend instance stores:

1. Calculation settings (method/basis/spin/nstates, etc.).
2. Current geometry.
3. Cached electronic-structure intermediates/results for that geometry.

Dynamics loop example::

    backend = PySCFBackend(params)
    for step in dynamics:
        backend.set_geom(propagate(...))   # invalidates geometry cache
        eg  = backend.energy_gradient(state_id=0)
        wfn = backend.wavefunction(state_id=0)
        nac = backend.analytical_nacme(state_id=0)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


# ═══════════════════════════════════════════════════════════════════════
#  Request (input)
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class MolecularGeometry:
    """Immutable snapshot of a molecular geometry.

    Attributes
    ----------
    atom_labels : list[str]
        Element symbols, length *natoms*.
    coords_bohr : np.ndarray
        Cartesian coordinates in **bohr**, shape ``(natoms, 3)``.
    """
    atom_labels: List[str]
    coords_bohr: np.ndarray        # (natoms, 3)

    def __post_init__(self):
        c = np.asarray(self.coords_bohr, dtype=np.float64)
        if c.ndim != 2 or c.shape[1] != 3:
            raise ValueError(f"coords_bohr must be (natoms, 3), got {c.shape}")
        if len(self.atom_labels) != c.shape[0]:
            raise ValueError("len(atom_labels) != number of coordinate rows")
        # frozen=True prevents assignment, so we use __dict__ for the cast
        object.__setattr__(self, "coords_bohr", c)


@dataclass(frozen=True)
class ESParams:
    """Method-agnostic calculation parameters (no geometry).

    Attributes
    ----------
    nstates : int
        Number of electronic states (including ground state).
    charge : int
        Total molecular charge.
    spin_multiplicity : int
        2S + 1 (1 = singlet, 3 = triplet, …).
    basis : str
        Basis set name recognised by the backend (e.g. ``"sto-3g"``).
    """
    nstates: int = 1
    charge: int = 0
    spin_multiplicity: int = 1
    basis: str = "sto-3g"


# ═══════════════════════════════════════════════════════════════════════
#  Results (output)
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class EnergyGradientResult:
    """Energies and nuclear gradients.

    Attributes
    ----------
    energies : np.ndarray
        Total electronic energies in **hartree**, shape ``(nstates,)``.
    gradients : np.ndarray or None
        Nuclear gradients in **hartree / bohr**, shape ``(nstates, natoms, 3)``.
        ``None`` when gradients were not requested.
    """
    energies: np.ndarray            # (nstates,)
    gradients: Optional[np.ndarray] = None  # (nstates, natoms, 3)


@dataclass(frozen=True)
class NACMEResult:
    """Nonadiabatic coupling matrix elements.

    Attributes
    ----------
    nacme : np.ndarray
        NACMEs in **hartree / bohr**, shape ``(nstates, nstates, natoms, 3)``.
    """
    nacme: np.ndarray               # (nstates, nstates, natoms, 3)


@dataclass(frozen=True)
class WavefunctionResult:
    """Wavefunction quantities from a variational CI calculation.

    All arrays assume a **shared MO basis** across states (valid for
    HF-based CI, SA-CASSCF, etc.).

    Attributes
    ----------
    mo_coefficients : np.ndarray
        MO-LCAO coefficient matrix, shape ``(nao, nmo)``.
    ao_overlap : np.ndarray
        AO overlap matrix *S*\\ :sub:`μν`, shape ``(nao, nao)``.
    ci_coefficients : np.ndarray
        CI eigenvectors, shape ``(ndet, nstates)``.
        Column *I* is the CI vector of state *I*.
    overlap : np.ndarray or None
        Inter-timestep state-overlap matrix ⟨Ψ_I(t)|Ψ_J(t′)⟩,
        shape ``(nstates, nstates)``.  Filled by the backend only when
        a previous-timestep wavefunction is available for comparison.
    """
    mo_coefficients: np.ndarray      # (nao, nmo)
    ao_overlap: np.ndarray           # (nao, nao)
    ci_coefficients: np.ndarray      # (ndet, nstates)
    overlap: Optional[np.ndarray] = None  # (nstates, nstates)


# ═══════════════════════════════════════════════════════════════════════
#  ESBackend – stateful backend bound to one current geometry
# ═══════════════════════════════════════════════════════════════════════

class ESBackend(ABC):

    @abstractmethod
    def set_geom(self, geometry: MolecularGeometry) -> None:
        """Set current geometry and invalidate geometry-dependent cache."""

    @abstractmethod
    def energy_gradient(self, state_id: int) -> EnergyGradientResult:
        """Energy and nuclear gradient data for the requested electronic state."""

    @abstractmethod
    def wavefunction(self, state_id: int) -> WavefunctionResult:
        """Wavefunction data for the requested electronic state."""

    @abstractmethod
    def analytical_nacme(self, state_id: int) -> Optional[NACMEResult]:
        """Analytical NACME data involving the requested state, or ``None``."""
