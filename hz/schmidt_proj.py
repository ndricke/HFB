import numpy as np
import scipy.linalg as slg
from frankenstein import lattice, scf, be
from frankenstein.tools.tensor_utils import mutate_all
from frankenstein.tools.sd_utils import xform_2, xform_4, xform_core, \
    eigh_reverse, _complementary_fragsites
from frankenstein.tools.io_utils import dumpMat
from frankenstein.tools.lat_utils import get_random_hV

from hf_proj import get_dV, get_ABD


def schmidt_decomposition_dm(P, fragsites, thresh_env=1.E-10):
    """P = [[Pimp, Pc.T], [Pc, Penv]]

    """
    nbas = P.shape[0]
    nimp = len(fragsites)
    cfragsites = _complementary_fragsites(nbas, fragsites)
    nc = len(cfragsites)

    Pimp = np.zeros([nimp, nimp])
    for i, fi in enumerate(fragsites):
        for j, fj in enumerate(fragsites):
            Pimp[i, j] = P[fi, fj]
    eimp, uimp = eigh_reverse(Pimp)
    A = uimp@np.diag(eimp**0.5)

    Pc = np.zeros([nc, nimp])
    for i, fi in enumerate(cfragsites):
        for j, fj in enumerate(fragsites):
            Pc[i, j] = P[fi, fj]
    B = Pc@uimp@np.diag(eimp**-0.5)

    if nc > 0:
        Penv = np.zeros([nc, nc])
        for i, fi in enumerate(cfragsites):
            for j, fj in enumerate(cfragsites):
                Penv[i, j] = P[fi, fj]
        DD = Penv - B@B.T
        eenv, uenv = eigh_reverse(DD)
        D = []
        for i in range(DD.shape[0]):
            if eenv[i] > thresh_env:
                D.append(uenv[:, i] * eenv[i]**0.5)
        nenv = len(D)
        D = np.array(D)
        TE = np.zeros([nbas, nenv])
        TE[cfragsites, :] = D.T
    else:
        TE = np.array([])

    B = B@np.diag((np.diag(B.T@B))**-0.5)
    T = np.zeros([nbas, nimp*2])
    T[fragsites, :nimp] = np.eye(nimp)
    T[cfragsites, nimp:] = B

    return T, TE


if __name__ == "__main__":
    nbas = 6
    nocc = 3

    np.random.seed(17)

    h, V = get_random_hV(nbas)
    lat = lattice.LATTICE(h=h, V=V, nocc=nocc)

    # get random dV
    udim = nbas*(nbas-1) + nbas*nbas
    u = np.random.rand(udim)
    dV = get_dV(u, nbas)

    mf = scf.RHF(lat)
    mf.kernel(verbose="mute")

    A, B = 1, 2
    h = np.random.rand(nbas, nbas)
    h += h.T
    # N -> [A, B] -> [A]
    T_AB, TE_AB = schmidt_decomposition_dm(mf.rdm1, [A, B])
    P_AB = T_AB.T @ mf.rdm1 @ T_AB
    T_AB_A, TE_AB_A = schmidt_decomposition_dm(P_AB, [0])
    h_AB_A = T_AB_A.T @ T_AB.T @ h @ T_AB @ T_AB_A

    # N -> [A]
    T_A, TE_A = schmidt_decomposition_dm(mf.rdm1, [A])
    h_A = T_A.T @ h @ T_A

#    dumpMat(h_AB_A, "N->[A,B]->[A]")
#    dumpMat(h_A, "N->[A]")

    dumpMat(T_AB)
    dumpMat(TE_AB)
    dumpMat(T_A)
    dumpMat(TE_A)

