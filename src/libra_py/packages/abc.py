"""Core interface definitions for electronic structure backends.

This module defines the minimal contract that all backend implementations
must satisfy. The base interface is stateless: all mutable state (caches,
computed intermediates, etc.) must be stored in concrete backend
implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class MolecularGeometry:
    atom_labels: list[str]
    coords_bohr: np.ndarray


@dataclass
class NACMEMatrix:
    matrix: np.ndarray


class IBackEnd(ABC):
    """Base interface for electronic structure backends.

    Implementations are allowed to store any backend-specific state internally.

    The interface is intentionally minimal: it only describes the operations
    required by a consumer (e.g. NAMD) and does not prescribe how a backend
    achieves those operations.
    """

    @abstractmethod
    def set_geom_and_run_hf(self, geom: MolecularGeometry) -> None:
        """Set the current geometry and run HF on it (storing results)."""

    @abstractmethod
    def compute_energy(self, root: int) -> float: #singlets only
        """Compute the energy for a given root."""

    @abstractmethod
    def compute_gradient(self, root: int) -> np.ndarray:
        """Compute the nuclear gradient for a given root."""

    @abstractmethod
    def time_overlap_matrix(self, nroots: int) -> np.ndarray:   
         """Compute the time-overlap matrix for the specified number of roots."""
