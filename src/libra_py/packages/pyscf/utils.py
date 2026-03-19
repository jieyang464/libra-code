import numpy as np
from dataclasses import dataclass
import numpy as np

@dataclass
class DetCIWavefunction:
    mo_coeff: np.ndarray      # (nao1, nmo1) or (nao2, nmo2)
    ci_coeff: np.ndarray      # (ndet,)
    determinants: list        # list[list[int]]
    ao_overlap: np.ndarray    # self-overlap of this WFN's AO basis (optional use)
    

def ci_overlap(wfn1: DetCIWavefunction, wfn2: DetCIWavefunction) -> float:
    """
    Compute overlap between two determinant-based CI wavefunctions.

    Each wavefunction provides:
        mo_coeff      : (nao, nmo)
        ci_coeff      : (ndet,)
        determinants  : list of occupied orbital index lists
        ao_overlap    : AO overlap used to connect the two MO bases

    Returns
    -------
    float
        <Psi1 | Psi2>
    """

    C1 = wfn1.mo_coeff
    C2 = wfn2.mo_coeff
    S_ao = wfn1.ao_overlap   # assumed cross AO overlap

    ci1 = np.asarray(wfn1.ci_coeff)
    ci2 = np.asarray(wfn2.ci_coeff)

    dets1 = wfn1.determinants
    dets2 = wfn2.determinants

    # Precompute full MO overlap matrix
    S_mo = C1.T @ S_ao @ C2

    overlap = 0.0

    for i, d1 in enumerate(dets1):
        for j, d2 in enumerate(dets2):

            # occupied submatrix
            S_occ = S_mo[np.ix_(d1, d2)]

            det_ovlp = np.linalg.det(S_occ)

            # For restricted (closed-shell) determinants, the spatial orbital
            # overlap contributes twice (alpha and beta spin blocks).
            det_ovlp = det_ovlp**2

            overlap += ci1[i] * ci2[j] * det_ovlp

    return float(overlap)


def ci_overlap_matrix(wfns1, wfns2):
    """Compute an overlap matrix between lists of wavefunctions.

    Parameters
    ----------
    wfns1 : Sequence[DetCIWavefunction]
    wfns2 : Sequence[DetCIWavefunction]

    Returns
    -------
    numpy.ndarray
        Overlap matrix (len(wfns1), len(wfns2)).
    """

    return np.array([[ci_overlap(w1, w2) for w2 in wfns2] for w1 in wfns1])
