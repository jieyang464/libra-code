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
.. module:: ci
   :platform: Unix, Windows
   :synopsis: this module implements functions for computing ci-level time-overlaps
.. moduleauthor:: Alexey V. Akimov, ChatGPT

"""

import numpy as np
from . import interfaces


def _build_spin_orbital_overlap(mo1, mo2, ao_overlap):
    """Build a spin-orbital MO overlap matrix between two MO sets.

    Parameters
    ----------
    mo1, mo2 : array-like
        Orbital coefficient matrices in the same AO basis (nao x nmo).
    ao_overlap : array-like
        AO overlap matrix (nao x nao).

    Returns
    -------
    ndarray
        Spin-orbital overlap matrix of shape (2*nmo, 2*nmo), built as
        kron(I_2, C1^T S_AO C2).
    """
    mo1 = np.asarray(mo1)
    mo2 = np.asarray(mo2)
    ao_overlap = np.asarray(ao_overlap)

    if mo1.shape[0] != ao_overlap.shape[0] or mo2.shape[0] != ao_overlap.shape[1]:
        raise ValueError("MO coefficient matrices must be compatible with AO overlap")

    S_mo = mo1.T @ ao_overlap @ mo2
    return np.kron(np.eye(2, dtype=S_mo.dtype), S_mo)


def _wavefunction_to_ci_data(wfn):
    """Convert a wavefunction-like object to the internal CI data format.

    The `overlap()` function expects data objects in the form:
        data[1] : list of lists of configurations for each excited state
        data[2] : list of lists of CI amplitudes for each excited state

    This helper assumes the wavefunction object provides:
        - `configurations`: list-of-list of determinants for excited states
        - `ci_coefficients`: array-like of shape (nconf, nstates) or (nconf,)

    Raises
    ------
    ValueError
        If the required attributes are not present.
    """
    if not hasattr(wfn, "configurations") or not hasattr(wfn, "ci_coefficients"):
        raise ValueError("Wavefunction object must have 'configurations' and 'ci_coefficients'")

    configs = wfn.configurations
    ci = np.asarray(wfn.ci_coefficients)

    if configs is None:
        raise ValueError("Wavefunction object is missing 'configurations' data")

    if ci.ndim == 1:
        ci_list = [ci.tolist()]
    elif ci.ndim == 2:
        # If ci has a ground-state column, drop it (common in some formats)
        if ci.shape[1] == len(configs) + 1:
            ci_list = [ci[:, i].tolist() for i in range(1, ci.shape[1])]
        else:
            ci_list = [ci[:, i].tolist() for i in range(ci.shape[1])]
    else:
        raise ValueError("ci_coefficients must be a 1D or 2D array")

    return [[], configs, ci_list]


def overlap_from_wavefunctions(wfn1, wfn2, params):
    """Compute CI-state overlaps from two determinant-based wavefunctions.

    This convenience wrapper builds the required `st_mo` and `data` structures
    and then calls :func:`overlap`.

    Parameters
    ----------
    wfn1, wfn2 : objects
        Wavefunction-like objects providing:
            - mo_coefficients
            - ao_overlap
            - configurations
            - ci_coefficients
    params : dict
        Same parameters as for :func:`overlap`.

    Returns
    -------
    ndarray
        CI overlap matrix.
    """
    if not hasattr(wfn1, "ao_overlap") or not hasattr(wfn2, "ao_overlap"):
        raise ValueError("Wavefunction objects must provide 'ao_overlap'")

    if not np.allclose(wfn1.ao_overlap, wfn2.ao_overlap):
        raise ValueError("AO overlap matrices must match between wavefunctions")

    st_mo = _build_spin_orbital_overlap(
        wfn1.mo_coefficients, wfn2.mo_coefficients, wfn1.ao_overlap
    )

    data1 = _wavefunction_to_ci_data(wfn1)
    data2 = _wavefunction_to_ci_data(wfn2)

    return overlap(st_mo, data1, data2, params)


def overlap(st_mo, data1, data2, params):
    """
    Compute the CI-state overlap matrix between two electronic-structure
    datasets using molecular-orbital time overlaps.

    This routine:
      1. Builds a common Slater-determinant basis from excited-state
         configurations of both datasets
      2. Constructs CI coefficient matrices in that common basis
      3. Computes SD and CSF overlap matrices (singlet)
      4. Projects the overlaps into the CI-state representation

    Parameters
    ----------
    st_mo : sparse matrix or array-like, shape (2*norb, 2*norb)
        Molecular-orbital time-overlap matrix in the spin–orbital basis.

    data1, data2 : tuple or list
        Electronic-structure data containers with the following layout::

            dataX[1] : list of list
                State-resolved configuration lists for excited states.
                dataX[1][i] contains configurations for excited state i+1.

            dataX[2] : list of list
                Corresponding CI amplitudes.
                dataX[2][i][j] is the amplitude of configuration j in
                excited state i+1.

        The ground state is assumed to be a pure reference determinant
        and is not included explicitly.

    params : dict
        Dictionary of required parameters:
            homo_indx : int
                HOMO index (1-based spatial orbital index)
            nocc : int
                Number of occupied orbitals below HOMO included in the window
            nvirt : int
                Number of virtual orbitals above HOMO included in the window
            nelec : int
                Total number of electrons
            nstates : int
                Total number of electronic states, including the ground state
            active_space : iterable of int, optional
                Spatial orbital indices (1-based) defining the active space for
                spin adaptation. If None, all available orbitals are used.

    Returns
    -------
    numpy.ndarray
        CI-state overlap matrix of shape `(nstates, nstates)`.

    Notes
    -----
    - This routine is restricted to closed-shell singlet states.
    - Only singly excited configurations are assumed.
    - The CI overlap is computed as::

          S_CI = C1ᵀ · S_CSF · C2

      where `C1` and `C2` are CI coefficient matrices in a common CSF basis.
    """
    # ------------------------------------------------------------------
    # Extract and validate parameters
    # ------------------------------------------------------------------
    required_keys = {"homo_indx", "nocc", "nvirt", "nelec", "nstates"}
    missing = required_keys - params.keys()
    if missing:
        raise KeyError(f"Missing required parameters: {missing}")

    homo_indx = params["homo_indx"]
    nocc = params["nocc"]
    nvirt = params["nvirt"]
    nelec = params["nelec"]
    nstates = params["nstates"]
    active_space = params["active_space"]

    if nstates <= 1:
        raise ValueError("nstates must include at least one excited state")

    # Orbital window (1-based spatial indices)
    lowest_orbital = homo_indx - nocc
    highest_orbital = homo_indx + nvirt

    # ------------------------------------------------------------------
    # Build common SD basis from excited states only
    # ------------------------------------------------------------------
    n_excited_states = nstates - 1

    common_sd_basis = interfaces.unique_confs(
        data1[1], data2[1], n_excited_states
    )

    # ------------------------------------------------------------------
    # CI coefficient matrices in the common SD basis
    # ------------------------------------------------------------------
    C1 = interfaces.ci_amplitudes_mtx(
        nstates, common_sd_basis, data1[1], data1[2]
    )
    C2 = interfaces.ci_amplitudes_mtx(
        nstates, common_sd_basis, data2[1], data2[2]
    )

    #-------------------------------------------------------------------
    # Get the effective number of electrons - only those that are in the
    # active space
    #-------------------------------------------------------------------
    nelec_eff = 0
    if active_space is None:
        nelec_eff = nelec
    else:
        nelec_eff = 2 * sum(i <= homo_indx for i in active_space)

    # ------------------------------------------------------------------
    # SD and CSF overlap matrices (singlet)
    # ------------------------------------------------------------------
    csf_ovlp, sd_ovlp = interfaces.sd_and_csf_overlaps_singlet(
        st_mo,
        lowest_orbital,
        highest_orbital,
        nelec_eff,
        homo_indx,
        common_sd_basis,
        active_space
    )

    # ------------------------------------------------------------------
    # CI-state overlap matrix
    # ------------------------------------------------------------------
    st_ci = C1.T @ csf_ovlp @ C2

    return st_ci

