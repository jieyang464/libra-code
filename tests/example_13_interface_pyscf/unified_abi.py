# *********************************************************************************
# * Copyright (C) 2026 Alexey V. Akimov
# *
# * This file is distributed under the terms of the GNU General Public License
# * as published by the Free Software Foundation, either version 3 of
# * the License, or (at your option) any later version.
# * See the file LICENSE in the root directory of this distribution
# * or <http://www.gnu.org/licenses/>.
# *
# *********************************************************************************/
"""
.. module:: unified_abi
   :platform: Unix, Windows
   :synopsis: Extended unified ABI for electronic-structure backends with wavefunction data

This module extends the base abi.py with comprehensive support for:
- Wavefunction data (CI coefficients, MO coefficients, AO overlaps)
- CASSCF/CASCI settings (active space, state averaging, weights)
- Basis set specifications
- Multi-configurational methods

Backwards compatible with the base ElectronicStructureStrategy interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Import base ABI classes for backwards compatibility
from libra_py.packages.es.abi import (
    ElectronicStructureRequest,
    ElectronicStructureResult,
    ElectronicStructureStrategy,
    GradientResult,
    SinglePointResult,
)


class MethodType(Enum):
    """Electronic structure method types."""

    HF = "hf"
    DFT = "dft"
    TDDFT = "tddft"
    CIS = "cis"
    CISD = "cisd"
    CASSCF = "casscf"
    CASCI = "casci"
    MRCI = "mrci"
    CC = "cc"
    MP2 = "mp2"
    SEMIEMPIRICAL = "semiempirical"
    CUSTOM = "custom"


class BasisSetType(Enum):
    """Basis set specification types."""

    STANDARD = "standard"  # e.g., "6-31G*", "cc-pVDZ"
    CUSTOM = "custom"  # Custom basis definition
    MINIMAL = "minimal"  # Minimal basis (e.g., STO-3G)
    NONE = "none"  # For semi-empirical or no basis


@dataclass(frozen=True)
class BasisSetSpecification:
    """
    Basis set specification for quantum chemistry calculations.

    Attributes:
        basis_type: Type of basis set specification
        basis_name: Standard basis set name (e.g., "6-31G*", "cc-pVDZ", "def2-TZVP")
        custom_basis_file: Path to custom basis set file (for CUSTOM type)
        ecp: Effective core potential specification (optional)
        auxiliary_basis: Auxiliary basis for RI/density fitting (optional)
    """

    basis_type: BasisSetType = BasisSetType.STANDARD
    basis_name: str = ""
    custom_basis_file: Optional[str] = None
    ecp: Optional[str] = None
    auxiliary_basis: Optional[str] = None


@dataclass(frozen=True)
class ActiveSpaceSpecification:
    """
    Active space specification for CASSCF/CASCI/MRCI calculations.

    Attributes:
        n_orbitals: Number of active orbitals (e.g., for CAS(6,6), n_orbitals=6)
        n_electrons: Number of active electrons (e.g., for CAS(6,6), n_electrons=6)
        orbital_indices: Specific orbital indices (1-based) to include in active space
                        If None, uses n_orbitals lowest unoccupied orbitals
        orbital_symmetries: Orbital symmetries for each active orbital (optional)
    """

    n_orbitals: int = 0
    n_electrons: int = 0
    orbital_indices: Optional[List[int]] = None
    orbital_symmetries: Optional[List[str]] = None

    def __post_init__(self):
        if self.n_orbitals < 0 or self.n_electrons < 0:
            raise ValueError("Active space dimensions must be non-negative")
        if self.n_electrons > 2 * self.n_orbitals:
            raise ValueError("Too many electrons for given active space")


@dataclass(frozen=True)
class StateAveragingSpecification:
    """
    State-averaging specification for multi-state calculations.

    Attributes:
        state_weights: Weights for state averaging [nstates_avg]
                      If None, equal weights for all states
        target_multiplicity: Target spin multiplicity for averaged states
        target_irrep: Target irreducible representation (point group symmetry)
    """

    state_weights: Optional[List[float]] = None
    target_multiplicity: Optional[int] = None
    target_irrep: Optional[str] = None


@dataclass(frozen=True)
class WavefunctionData:
    """
    Wavefunction data output from electronic structure calculations.

    All matrices use standard quantum chemistry conventions:
    - AO basis: atomic orbital basis (typically Gaussian basis functions)
    - MO basis: molecular orbital basis
    - CI basis: configuration interaction basis

    Attributes:
        mo_coefficients: MO-LCAO coefficients [nao, nmo] or [nao, nact] for active space
        ao_overlap: AO overlap matrix S_μν [nao, nao]
        ci_coefficients: CI coefficients [nconf, nstates] or [ndet, nstates]
        ci_energies: CI state energies [nstates] (redundant with energies, but included for completeness)
        orbital_energies: Orbital energies (eigenvalues) [nmo] or [nact]
        occupation_numbers: Orbital occupation numbers [nmo] or [nact]
        transition_dipoles: Transition dipole moments [nstates, nstates, 3] (optional)
        configurations: Configuration/determinant descriptions (optional, format depends on method)
        spin_density: Spin density matrix in AO basis [nao, nao] (optional, for open-shell)
        natural_orbitals: Natural orbital coefficients [nao, nmo] (optional)
        natural_occupations: Natural orbital occupations [nmo] (optional)
    """

    mo_coefficients: Optional[np.ndarray] = None
    ao_overlap: Optional[np.ndarray] = None
    ci_coefficients: Optional[np.ndarray] = None
    ci_energies: Optional[np.ndarray] = None
    orbital_energies: Optional[np.ndarray] = None
    occupation_numbers: Optional[np.ndarray] = None
    transition_dipoles: Optional[np.ndarray] = None
    configurations: Optional[Any] = None
    spin_density: Optional[np.ndarray] = None
    natural_orbitals: Optional[np.ndarray] = None
    natural_occupations: Optional[np.ndarray] = None


@dataclass(frozen=True)
class EnhancedElectronicStructureRequest(ElectronicStructureRequest):
    """
    Enhanced ES request with support for multi-configurational methods.

    Extends the base ElectronicStructureRequest with:
    - Detailed method specification
    - Basis set information
    - Active space and state averaging for CASSCF/CASCI
    - Wavefunction output requests

    Attributes:
        method_type: Type of electronic structure method
        method_details: Method-specific details (e.g., functional name for DFT)
        basis_set: Basis set specification
        active_space: Active space specification for CASSCF/CASCI/MRCI
        state_averaging: State averaging specification
        request_wavefunction: Whether to compute and return wavefunction data
        request_nacme: Whether to compute nonadiabatic coupling matrix elements
        request_transition_dipoles: Whether to compute transition dipole moments
        convergence_threshold: SCF/optimization convergence threshold
        max_iterations: Maximum number of SCF/optimization iterations
    """

    method_type: MethodType = MethodType.HF
    method_details: str = ""
    basis_set: Optional[BasisSetSpecification] = None
    active_space: Optional[ActiveSpaceSpecification] = None
    state_averaging: Optional[StateAveragingSpecification] = None
    request_wavefunction: bool = False
    request_nacme: bool = False
    request_transition_dipoles: bool = False
    convergence_threshold: float = 1e-8
    max_iterations: int = 100


@dataclass(frozen=True)
class EnhancedElectronicStructureResult(ElectronicStructureResult):
    """
    Enhanced ES result including wavefunction data.

    Extends the base ElectronicStructureResult with wavefunction information
    needed for NAMD and multi-configurational analysis.

    Attributes:
        wavefunction: Wavefunction data (MO coeffs, CI coeffs, overlaps)
        nacme: Nonadiabatic coupling matrix elements [nstates, nstates, ndof]
        transition_dipoles: Transition dipole matrix [nstates, nstates, 3]
        dipole_moments: State dipole moments [nstates, 3]
        oscillator_strengths: Oscillator strengths [nstates] (for excited states)
        mulliken_charges: Mulliken atomic charges [natoms, nstates]
    """

    wavefunction: Optional[WavefunctionData] = None
    nacme: Optional[np.ndarray] = None
    transition_dipoles: Optional[np.ndarray] = None
    dipole_moments: Optional[np.ndarray] = None
    oscillator_strengths: Optional[np.ndarray] = None
    mulliken_charges: Optional[np.ndarray] = None


class UnifiedElectronicStructureStrategy(ElectronicStructureStrategy):
    """
    Extended ABC for unified ES backends with wavefunction support.

    Implementations should:
    1. Accept EnhancedElectronicStructureRequest (or base ElectronicStructureRequest)
    2. Return EnhancedElectronicStructureResult with requested data
    3. Handle active space and state averaging for multi-configurational methods
    4. Compute wavefunction data when requested
    """

    @abstractmethod
    def evaluate_extended(
        self, request: EnhancedElectronicStructureRequest
    ) -> EnhancedElectronicStructureResult:
        """
        Unified evaluation with full wavefunction output.

        Args:
            request: Enhanced request with method/basis/active space details

        Returns:
            Enhanced result including energies, gradients, and wavefunction data

        Raises:
            NotImplementedError: If method/basis combination is not supported
            ValueError: If active space or request parameters are invalid
        """

    def evaluate(
        self, request: ElectronicStructureRequest
    ) -> ElectronicStructureResult:
        """
        Backwards-compatible evaluation (base ABI).

        Default implementation converts to EnhancedRequest and calls evaluate_extended.
        """
        # Convert base request to enhanced request with defaults
        enhanced_request = EnhancedElectronicStructureRequest(
            geometry_bohr=request.geometry_bohr,
            atom_labels=request.atom_labels,
            nstates=request.nstates,
            grad_method_gs=request.grad_method_gs,
            grad_method_ex=request.grad_method_ex,
            charge=request.charge,
            spin_multiplicity=request.spin_multiplicity,
            options=request.options,
            backend_params=request.backend_params,
            verbosity=request.verbosity,
            traj_id=request.traj_id,
            step=request.step,
            output_file=request.output_file,
        )

        enhanced_result = self.evaluate_extended(enhanced_request)

        # Return base result (drop wavefunction data)
        return ElectronicStructureResult(
            energies_hartree=enhanced_result.energies_hartree,
            gradients_hartree_per_bohr=enhanced_result.gradients_hartree_per_bohr,
            adiabatic_vectors=enhanced_result.adiabatic_vectors,
        )

    def nacme(self, request: ElectronicStructureRequest) -> Optional[np.ndarray]:
        """
        Compute NACME via evaluate_extended.

        Default implementation upgrades request and extracts NACME from result.
        """
        enhanced_request = EnhancedElectronicStructureRequest(
            geometry_bohr=request.geometry_bohr,
            atom_labels=request.atom_labels,
            nstates=request.nstates,
            grad_method_gs=request.grad_method_gs,
            grad_method_ex=request.grad_method_ex,
            charge=request.charge,
            spin_multiplicity=request.spin_multiplicity,
            options=request.options,
            backend_params=request.backend_params,
            verbosity=request.verbosity,
            traj_id=request.traj_id,
            step=request.step,
            output_file=request.output_file,
            request_nacme=True,
        )

        result = self.evaluate_extended(enhanced_request)
        return result.nacme


# Convenience type aliases
WfnData = WavefunctionData
CASSettings = ActiveSpaceSpecification
UnifiedStrategy = UnifiedElectronicStructureStrategy
