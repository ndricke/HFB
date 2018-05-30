"""
This script tests performing Schmidt decomposition on a HFB wave function.
"""


import numpy as np
import sys

import hfbSdEval

#from hfb_utils_sp import hfb_solver_sp

from frankenstein import scf, be
from frankenstein.tools.io_utils import dumpMat, dumpVec
from frankenstein.tools.lat_utils import get_Hubbard, get_Hubbard_lr
from frankenstein.tools.sd_utils import _complementary_fragsites, \
    eigh_reverse, schmidt_decomposition, xform_core, xform_2, xform_4
from frankenstein.tools.hfb_utils import get_hfb_Rs, get_hfb_energy, get_hfb_PKs


def pad_matrix(A, nrep, inds, signs):
    n = A.shape[0]

    As = np.zeros([n*nrep, n*nrep])
    for ind, sign in zip(inds, signs):
        i, j = ind
        As[i*n:(i+1)*n, j*n:(j+1)*n] = A*sign

    return As


def pad_tensor(V, nrep, inds, signs):
    n = V.shape[0]

    Vs = np.zeros([n*nrep, n*nrep, n*nrep, n*nrep])
    for ind, sign in zip(inds, signs):
        i, j, k, l = ind
        Vs[i*n:(i+1)*n, j*n:(j+1)*n, k*n:(k+1)*n, l*n:(l+1)*n] = sign*V

    return Vs


def ABs_split(ABs, fragsites):
    nbas, nimp = ABs[0].shape[0]//2, ABs[0].shape[1]
    assert(nimp == len(fragsites))
    nimp0 = nimp//2

    ABs_spl = []
    for s in [0, 1]:
        AB = ABs[s]
        AB_spl = np.zeros([2*nbas, 2*nimp])
        for i in range(2*nbas):
            if i in fragsites:
                AB_spl[i, :nimp] = AB[i, :nimp]
            else:
                AB_spl[i, nimp:2*nimp] = AB[i, :nimp]
        AB_spl = AB_spl @ np.diag((np.diag(AB_spl.T@AB_spl))**-0.5)
        ABs_spl.append(AB_spl)

    return ABs_spl


def get_hfb_SD_dm(Rs, fragsites0, thresh_env=1.E-7, split=False):
    """Performing Schmidt decomposition for a HFB wave function.
    """
    nbas = Rs[0].shape[0]//2

    nimp0 = len(fragsites0)
    nimp = nimp0*2
    ncore = nbas-nimp
    fragsites = fragsites0 + [fgs0+nbas for fgs0 in fragsites0]
    # cfragsites = _complementary_fragsites(2*nbas, fragsites)

    ABs, Ds = [], []
    for s in [0, 1]:
        R = Rs[s]
        Rimp = np.zeros([nimp, nimp])
        for i, fi in enumerate(fragsites):
            for j, fj in enumerate(fragsites):
                Rimp[i, j] = R[fi, fj]
        eimp, Wimp = eigh_reverse(Rimp)
        ATinv = Wimp @ np.diag(eimp**-0.5)
        ABs.append(R[:, fragsites] @ ATinv)

        Renv = R-ABs[s]@ABs[s].T
        eenv, Wenv = eigh_reverse(Renv)
        assert(sum(np.abs(eenv)>thresh_env)==ncore)
        Ds.append(Wenv[:, :ncore] @ np.diag(eenv[:ncore]**0.5))

    if split:
        ABs_spl = ABs_split(ABs, fragsites)
        return ABs, Ds, ABs_spl
    else:
        return ABs, Ds


def hfb_in_hfb_embedding(h_qp, V_qp, ABs, Ds):
    # form xform mat T^s = [[U^s, V^s], [V^t, U^t]]
    Ts = hfbSdEval.get_hfb_Ts(ABs)
    Renvs = [Ds[s]@Ds[s].T for s in [0, 1]]

    # Xform into Schmidt space
    hs_qp_sd, hcores_sd, Dcores_sd, Vs_qp_sd = [], [], [], []
    for s in [0, 1]:
        hcore = np.einsum("pqrs,sr->pq", V_qp, Renvs[s]+Renvs[1-s]) - \
            np.einsum("psrq,sr->pq", V_qp, Renvs[s])
        hcores_sd.append(xform_2(hcore, Ts[s]))

        Dcore = np.einsum("psqr,sr->pq", V_qp, Renvs[s])
        Dcores_sd.append(xform_2(Dcore, Ts[s]))

        hs_qp_sd.append(xform_2(h_qp, Ts[s])+hcores_sd[s]+Dcores_sd[s])
        Vs_qp_sd.append(xform_4(V_qp, Ts[s]))

        dumpMat(hs_qp_sd[s])

    for s in [0, 1]:
        Vs_qp_sd.append(xform_4(V_qp, Ts[s], Ts[1-s]))

    # Solve HFB-in-HFB embedding space
    # hfb_solver_sp()


if __name__ == "__main__":
    nbas = 4
    nocc = 2
    U = -3

    # get Hamiltonian
    h, V = get_Hubbard_lr(nbas, U, U*0.75, nocc=nocc)
    # V = np.zeros(V.shape)

    # read solution from file
    Ws0 = []
    for s, sstr in enumerate(["alpha", "beta"]):
        Ws0.append(np.loadtxt("hz/W_{:s}".format(sstr)))
    Rs0 = get_hfb_Rs(Ws0)
    # Rs0 = None

    # fake-run
    mf = scf.HFB(h=h, V=V, nocc=nocc, unrestricted=False, opt="damping", max_iter=10000, conv=10)
    mf.kernel(verbose="mute", Rs0=Rs0)

    print("  ### Benchmark ###")
    print("E(HF)    : % .10f" % mf.e_hf)
    print("E(Pair)  : % .10f" % mf.e_pr)
    print("E(HFB)   : % .10f" % mf.e_hfb)
    print()


    es, Ws, Rs, Hs = mf.qp_energy, mf.qp_coeff, mf.qp_rdm1, mf.qp_fock

    # SD
    fragsites0 = [2]
    ABs, Ds, ABs_spl = get_hfb_SD_dm(Rs, fragsites0, split=True)


    Ts = hfbSdEval.get_hfb_Ts(ABs)


    # check energy
    Eh_hf = pad_matrix(h, 2, [[0, 0]], [1])
    EV_hf = pad_tensor(V, 2, [[0,0,0,0]], [1])
    EV_pr = pad_tensor(V, 2, [[1,1,0,0], [0,0,1,1]], [0.5,0.5])

    ## original space
    hfbSdEval.energy_check_original_space(h, V, ABs, Rs)
    hfbSdEval.energy_check_original_space_quasiparticle(Eh_hf, EV_hf, EV_pr, ABs, Rs)

    ## Schmidt space
    # hfbSdEval.energy_check_schmidt_space(h, V, ABs, Ds, Rs)
    hfbSdEval.energy_check_schmidt_space_quasiparticle(Eh_hf, EV_hf, EV_pr, ABs, Ds, Rs)

    # HFB-in-HFB embedding
    h_qp = pad_matrix(h, 2, [[0, 0], [1, 1]], [1, -1])
    V_qp = pad_tensor(V, 2, [[0,0,0,0], [1,1,0,0], [0,0,1,1], [1,1,1,1]], \
        [1,-1,-1,1])
    hfb_in_hfb_embedding(h_qp, V_qp, ABs, Ds)

















    
