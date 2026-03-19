"""PySCF-based CASSCF backend for the universal ES interface."""

from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any, Optional, Tuple, Union, Sequence
import numpy as np
from pyscf import fci, gto, mcscf, scf
from pyscf.fci import cistring
from ..abc import IBackEnd, MolecularGeometry, NACMEMatrix
from . import utils


class CASSCFBackend(IBackEnd):
    """PySCF-based CASSCF backend for the universal ES interface."""

    # 1) when a geom is set the HF is run; the MF is written to the member attribute
    #    `_mf`
    # 2) when the CASSCF energy is requested for a certain root, the CASSCF is run
    #    (number of roots requested = self._nroots). The resulting MC object is
    #    stored in the member attribute `mc`, which also contains energies of other
    #    states.

    def __init__(
        self,
        mol: Optional[Any] = None,
        norbcas: int = 0,
        nelecas:int = 0, # can be either total number of electrons in the active space (for closed-shell case) or a tuple of (alpha, beta) electrons for open-shell case
        nroots: int = 1,
        basis: str = "sto-3g",
        unit: str = "Bohr",
        charge: int = 0,
    ) -> None:
        self._mol: Optional[Any] = mol
        self._norbcas: int = norbcas
        self._nelecas: Union[int, Tuple[int, int]] = nelecas  # must be even for closed-shell case or (alpha,beta)
        self._nroots: int = nroots
        # new: store basis and unit so geometry setup is configurable
        self._basis: str = basis
        self._unit: str = unit
        # overall molecular charge (user-specified); default 0
        self._charge: int = int(charge)
        self._mf: Optional[Any] = None
        self._mc: Optional[Any] = None
        self._wfncache: Optional[utils.DetCIWavefunction] = None  # wfn info from previous geom for time-overlap computation and casscf initial guess

    def set_geom_and_run_hf(self, geom: MolecularGeometry) -> None:
        prev_mol: Optional[Any] = self._mol

        # cache casscf wfn before reseting geometry and running HF
        if self._mc is not None:
            self._wfncache = utils.DetCIWavefunction(
                mo_coeff=self._mc.mo_coeff,
                ci_coeff=self._mc.ci,
                determinants=getattr(self._mc, "ci_dets", []),
                ao_overlap=None,
            )

        # resetting geom dependent attributes
        self._mc = None

        # Set up the molecule and run HF using configured basis/unit
        # Use the user-specified overall molecular charge when constructing the molecule.
        charge: int = self._charge
        self._mol = gto.M(
            atom=";".join(
                f"{label} {coord[0]} {coord[1]} {coord[2]}"
                for label, coord in zip(geom.atom_labels, geom.coords_bohr)
            ),
            basis=self._basis,
            unit=self._unit,
            charge=charge,
            spin=0,
        )
        if self._wfncache is not None and prev_mol is not None:
            self._wfncache.ao_overlap = gto.intor_cross("cint1e_ovlp_sph", prev_mol, self._mol)
        self._mf = scf.RHF(self._mol).run(verbose=0)
        
    def compute_energy(self, root: int) -> float:
        if self._mc is not None:
            e_states: Optional[Sequence[float]] = getattr(self._mc, "e_states", None)
            if e_states is not None:
                return float(np.asarray(e_states)[root])
            return float(self._mc.e_tot)

        # initialize CASSCF object if not present
        if self._mc is None:
            self._mc = mcscf.CASSCF(self._mf, self._norbcas, self._nelecas)
            self._mc.fcisolver = fci.direct_spin0.FCI(self._mol)
            self._mc.fcisolver.nroots = self._nroots
            if self._nroots > 1:
                self._mc = self._mc.state_average_([1.0 / self._nroots] * self._nroots)  # equally average over the requested number of roots

        if self._wfncache is not None and self._wfncache.mo_coeff is not None:
            self._mc.kernel(mo_coeff=self._wfncache.mo_coeff)  # read MO coeffs from cache to ensure active space stays the same
        else:
            self._mc.kernel(mo_coeff=self._mf.mo_coeff)  # no cache, run normally and save MO coeffs to

        e_states: Optional[Sequence[float]] = getattr(self._mc, "e_states", None)
        if e_states is not None:
            return float(np.asarray(e_states)[root])
        return float(self._mc.e_tot)

    def compute_gradient(self, root: int) -> np.ndarray:
        # restart geadient on root n from self._mc
        if self._mc is None:
            raise ValueError("CASSCF must be run before computing gradients.")
        e_states: Optional[Sequence[float]] = getattr(self._mc, "e_states", None)
        if e_states is not None:
            if root < 0 or root >= len(e_states):
                raise IndexError(f"Requested root {root}, but only {len(e_states)} roots are available.")
            return np.asarray(self._mc.nuc_grad_method(state=root).kernel())
        if root != 0:
            raise IndexError("Only root 0 is available for a single-state CASSCF calculation.")
        return np.asarray(self._mc.nuc_grad_method().kernel())
    
    def time_overlap_matrix(self, nroots: int) -> np.ndarray:
        # compute time-overlap matrix between current wfn and cached wfn (if available)
        # (if self._nroots=3 it should return a 3x3 matrix of overlaps between the first 3 roots)
        if self._mc is None:
            raise ValueError("CASSCF must be run before computing time-overlap matrix.")
        if self._wfncache is None:
            raise ValueError("No cached wavefunction available for time-overlap computation.")
        if nroots > self._nroots:
            raise ValueError(f"Requested {nroots} roots, but only {self._nroots} are available.")
        if self._wfncache.ao_overlap is None:
            raise ValueError("Cached AO overlap between consecutive geometries is not available.")

        ci_prev = self._wfncache.ci_coeff
        if not isinstance(ci_prev, (list, tuple)):
            ci_prev = [ci_prev]

        ci_curr = self._mc.ci
        if not isinstance(ci_curr, (list, tuple)):
            ci_curr = [ci_curr]

        ncore: int = int(getattr(self._mc, "ncore", 0))
        mo_prev_cas = self._wfncache.mo_coeff[:, ncore : ncore + self._norbcas]
        mo_curr_cas = self._mc.mo_coeff[:, ncore : ncore + self._norbcas]
        s_mo = mo_prev_cas.T.conj() @ self._wfncache.ao_overlap @ mo_curr_cas

        overlap: np.ndarray = np.zeros((nroots, nroots), dtype=np.complex128)
        for i in range(nroots):
            for j in range(nroots):
                overlap[i, j] = fci.addons.overlap(
                    np.asarray(ci_prev[i]),
                    np.asarray(ci_curr[j]),
                    self._norbcas,
                    self._nelecas,
                    s=s_mo,
                )

        return np.asarray(np.real_if_close(overlap))
