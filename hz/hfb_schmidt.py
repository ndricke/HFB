"""
This script tests performing Schmidt decomposition on a HFB wave function.
"""


import numpy as np
import sys

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


def energy_check_original_space(h, V, ABs, Rs):
    nbas = ABs[0].shape[0]//2
    Ps, Ks = get_hfb_PKs(Rs)

    # no embedding
    e1_hf, e2_hf, e_hf, e_pr = 0., 0., 0., 0.
    for s in [0, 1]:
        e1_hf += np.sum(h*Ps[s])
        e2_hf += 0.5*np.sum((np.einsum("pqrs,sr->pq", V, Ps[s]+Ps[1-s]) - \
            np.einsum("psrq,sr->pq", V, Ps[s]))*Ps[s])
    e_hf = e1_hf + e2_hf
    e_pr = np.sum(np.einsum("psqr,sr->pq", V, Ks[0])*Ks[0])

    # embedding
    Rimps = [ABs[s]@ABs[s].T for s in [0, 1]]
    Renvs = [Rs[s]-Rimps[s] for s in [0, 1]]
    Pimps = [Rimps[s][:nbas, :nbas] for s in [0, 1]]
    Penvs = [Renvs[s][:nbas, :nbas] for s in [0, 1]]
    Kimps = [Rimps[s][:nbas, nbas:2*nbas] for s in [0, 1]]
    Kenvs = [Renvs[s][:nbas, nbas:2*nbas] for s in [0, 1]]
    eimp_hf, eimp_pr, eenv_hf, eenv_pr, v_hf, v_pr = [0. for i in range(6)]
    for s in [0, 1]:
        eimp_hf += np.sum(h*Pimps[s]) + \
            0.5*np.sum((np.einsum("pqrs,sr->pq", V, Pimps[s]+Pimps[1-s]) - \
            np.einsum("psrq,sr->pq", V, Pimps[s]))*Pimps[s])
        v_hf += np.sum((np.einsum("pqrs,sr->pq", V, Penvs[s]+Penvs[1-s]) - \
            np.einsum("psrq,sr->pq", V, Penvs[s]))*Pimps[s])
        eenv_hf += np.sum(h*Penvs[s]) + \
            0.5*np.sum((np.einsum("pqrs,sr->pq", V, Penvs[s]+Penvs[1-s]) - \
            np.einsum("psrq,sr->pq", V, Penvs[s]))*Penvs[s])
    eimp_pr = np.sum(np.einsum("psqr,sr->pq", V, Kimps[0])*Kimps[0])
    v_pr = 2*np.sum(np.einsum("psqr,sr->pq", V, Kenvs[0])*Kimps[0])
    eenv_pr = np.sum(np.einsum("psqr,sr->pq", V, Kenvs[0])*Kenvs[0])

    # print out
    print("  ### HF (Site Basis) ###")
    print("E(full)    = % .10f" % e_hf)
    print("E(1)       = % .10f" % e1_hf)
    print("E(2)       = % .10f" % e2_hf)
    print("E(imp)     = % .10f" % eimp_hf)
    print("E(env)     = % .10f" % eenv_hf)
    print("V(imp-env) = % .10f" % v_hf)
    print("Sum        = % .10f" % (eimp_hf+eenv_hf+v_hf))
    print("Check      ? %s" % (np.allclose(e_hf, eimp_hf+eenv_hf+v_hf)))
    print()

    print("  ### Pair (Site Basis) ###")
    print("E(full)    = % .10f" % e_pr)
    print("E(imp)     = % .10f" % eimp_pr)
    print("E(env)     = % .10f" % eenv_pr)
    print("V(imp-env) = % .10f" % v_pr)
    print("Sum        = % .10f" % (eimp_pr+eenv_pr+v_pr))
    print("Check      ? %s" % (np.allclose(e_pr, eimp_pr+eenv_pr+v_pr)))
    print()


def energy_check_original_space_quasiparticle(Eh_hf, EV_hf, EV_pr, ABs, Rs):
    e_hf, e1_hf, e2_hf, e_pr = 0., 0., 0., 0.
    for s in [0, 1]:
        e1_hf += np.sum(Eh_hf*Rs[s])
        e2_hf += 0.5*np.sum((np.einsum("pqrs,sr->pq", EV_hf, Rs[s]+Rs[1-s]) - \
            np.einsum("psrq,sr->pq", EV_hf, Rs[s]))*Rs[s])
        e_pr += 0.5*np.sum(np.einsum("psqr,sr->pq", EV_pr, Rs[s])*Rs[s])
    e_hf = e1_hf + e2_hf
    print("  ### HF (Quasi-Particle Basis) ###")
    print("E(full)    = % .10f" % e_hf)
    print("E(1)       = % .10f" % e1_hf)
    print("E(2)       = % .10f" % e2_hf)
    print()

    print("  ### Pair (Quasi-Particle Basis) ###")
    print("E(full)    = % .10f" % e_pr)
    print()


def reverse_rows(AB):
    n = AB.shape[0]//2
    BA = np.zeros(AB.shape)
    BA[:n, :] = AB[-n:, :]
    BA[-n:, :] = AB[:n, :]

    return BA


def get_hfb_Ts(ABs):
    nbas = ABs[0].shape[0]//2
    nimp = ABs[0].shape[1]

    Ts = []
    for s in [0, 1]:
        T = np.zeros([2*nbas, 2*nimp])
        T[:, :nimp] = ABs[s]
        T[:, -nimp:] = reverse_rows(ABs[1-s])
        Ts.append(T)

    return Ts


def energy_check_schmidt_space_quasiparticle(Eh_hf, EV_hf, EV_pr, ABs, Ds, Rs):
    nbas = ABs[0].shape[0]//2
    nimp = ABs[0].shape[1]

    # form xform mat T^s = [[U^s, V^s], [V^t, U^t]]
    Ts = get_hfb_Ts(ABs)

    # Xform into Schmidt space
    Ehs_hf_sd, EVs_hf_sd, EVs_pr_sd = [], [], []
    for s in [0, 1]:
        Ehs_hf_sd.append(xform_2(Eh_hf, Ts[s]))
        EVs_hf_sd.append(xform_4(EV_hf, Ts[s]))
        EVs_pr_sd.append(xform_4(EV_pr, Ts[s]))

    for s in [0, 1]:
        EVs_hf_sd.append(xform_4(EV_hf, Ts[s], Ts[1-s]))
        EVs_pr_sd.append(xform_4(EV_pr, Ts[s], Ts[1-s]))

    # embedding
    Rimps = [ABs[s]@ABs[s].T for s in [0, 1]]
    # for s in [0, 1]:
    #     dumpMat(Rimps[s], "Rimps[{:d}] {:.10f}".format(s, \
    #     np.trace(Rimps[s][:nbas, :nbas])))
    Rimps_sd = [xform_2(Rimps[s], Ts[s]) for s in [0, 1]]
    # for s in [0, 1]:
    #     dumpMat(Rimps_sd[s], "Rimps_sd[{:d}] {:.10f}".format(s, \
    #     np.trace(Rimps_sd[s][:nimp, :nimp])))
    Renvs = [Rs[s]-Rimps[s] for s in [0, 1]]
    hcores = [np.einsum("pqrs,sr->pq", EV_hf, Renvs[s]+Renvs[1-s])-\
        np.einsum("psrq,sr->pq", EV_hf, Renvs[s]) for s in [0, 1]]
    hcores_sd = [xform_2(np.einsum("pqrs,sr->pq", EV_hf, Renvs[s]+Renvs[1-s])-\
        np.einsum("psrq,sr->pq", EV_hf, Renvs[s]), Ts[s]) for s in [0, 1]]
    Dcores = [np.einsum("psqr,sr->pq", EV_pr, Renvs[s]) for s in [0, 1]]
    Dcores_sd = [xform_2(np.einsum("psqr,sr->pq", EV_pr, Renvs[s]), Ts[s]) \
        for s in [0, 1]]
    eimp_hf, eenv_hf, v_hf, eimp_pr, eenv_pr, v_pr = 0., 0., 0., 0., 0., 0.
    for s in [0, 1]:
        eimp_hf += np.sum(Ehs_hf_sd[s]*Rimps_sd[s]) + \
            0.5*np.sum((np.einsum("pqrs,sr->pq", EVs_hf_sd[s], Rimps_sd[s]) - \
                np.einsum("psrq,sr->pq", EVs_hf_sd[s], Rimps_sd[s]) + \
                np.einsum("pqrs,sr->pq", EVs_hf_sd[s+2], Rimps_sd[1-s])) * \
                Rimps_sd[s])
        eenv_hf += np.sum((Eh_hf+0.5*hcores[s])*Renvs[s])
        v_hf += np.sum(hcores_sd[s] * Rimps_sd[s])
        eimp_pr += \
            0.5*np.sum(np.einsum("psqr,sr->pq", EVs_pr_sd[s], Rimps_sd[s]) * \
            Rimps_sd[s])
        eenv_pr += 0.5*np.sum(Dcores[s]*Renvs[s])
        v_pr += np.sum(Dcores_sd[s] * Rimps_sd[s])
    print("  ### HF (Schmidt Basis) ###")
    print("E(imp)     = % .10f" % eimp_hf)
    print("E(env)     = % .10f" % eenv_hf)
    print("V(imp-env) = % .10f" % v_hf)
    print("Sum        = % .10f" % (eimp_hf+eenv_hf+v_hf))
    print()

    print("  ### Pair (Schmidt Basis) ###")
    print("E(imp)     = % .10f" % eimp_pr)
    print("E(env)     = % .10f" % eenv_pr)
    print("V(imp-env) = % .10f" % v_pr)
    print("Sum        = % .10f" % (eimp_pr+eenv_pr+v_pr))
    print()


def hfb_in_hfb_embedding(h_qp, V_qp, ABs, Ds):
    # form xform mat T^s = [[U^s, V^s], [V^t, U^t]]
    Ts = get_hfb_Ts(ABs)
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
        Ws0.append(np.loadtxt("W_{:s}".format(sstr)))
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

    # check energy
    Eh_hf = pad_matrix(h, 2, [[0, 0]], [1])
    EV_hf = pad_tensor(V, 2, [[0,0,0,0]], [1])
    EV_pr = pad_tensor(V, 2, [[1,1,0,0], [0,0,1,1]], [0.5,0.5])

    ## original space
    energy_check_original_space(h, V, ABs, Rs)
    energy_check_original_space_quasiparticle(Eh_hf, EV_hf, EV_pr, ABs, Rs)

    ## Schmidt space
    # energy_check_schmidt_space(h, V, ABs, Ds, Rs)
    energy_check_schmidt_space_quasiparticle(Eh_hf, EV_hf, EV_pr, ABs, Ds, Rs)

    # HFB-in-HFB embedding
    h_qp = pad_matrix(h, 2, [[0, 0], [1, 1]], [1, -1])
    V_qp = pad_tensor(V, 2, [[0,0,0,0], [1,1,0,0], [0,0,1,1], [1,1,1,1]], \
        [1,-1,-1,1])
    hfb_in_hfb_embedding(h_qp, V_qp, ABs, Ds)

















    
