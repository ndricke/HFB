import numpy as np
import sys

from frankenstein import scf, be
from frankenstein.tools.io_utils import dumpMat, dumpVec
from frankenstein.tools.lat_utils import get_Hubbard, get_Hubbard_lr
from frankenstein.tools.sd_utils import _complementary_fragsites, \
    eigh_reverse, schmidt_decomposition, xform_core, xform_2, xform_4
from frankenstein.tools.hfb_utils import get_hfb_Rs, get_hfb_energy, get_hfb_PKs

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
    hcores = [np.einsum("pqrs,sr->pq", EV_hf, Renvs[s]+Renvs[1-s]) - np.einsum("psrq,sr->pq", EV_hf, Renvs[s]) for s in [0, 1]]
    hcores_sd = [xform_2(np.einsum("pqrs,sr->pq", EV_hf, Renvs[s]+Renvs[1-s]) - np.einsum("psrq,sr->pq", EV_hf, Renvs[s]), Ts[s]) for s in [0, 1]]
    Dcores = [np.einsum("psqr,sr->pq", EV_pr, Renvs[s]) for s in [0, 1]]
    Dcores_sd = [xform_2(np.einsum("psqr,sr->pq", EV_pr, Renvs[s]), Ts[s]) for s in [0, 1]]

    eimp_hf, eenv_hf, v_hf, eimp_pr, eenv_pr, v_pr = 0., 0., 0., 0., 0., 0.
    for s in [0, 1]:
        eimp_hf += np.sum(Ehs_hf_sd[s]*Rimps_sd[s]) + \
            0.5*np.sum((np.einsum("pqrs,sr->pq", EVs_hf_sd[s], Rimps_sd[s]) - \
                np.einsum("psrq,sr->pq", EVs_hf_sd[s], Rimps_sd[s]) + \
                np.einsum("pqrs,sr->pq", EVs_hf_sd[s+2], Rimps_sd[1-s])) * \
                Rimps_sd[s])

        eenv_hf += np.sum((Eh_hf+0.5*hcores[s])*Renvs[s])
        v_hf += np.sum(hcores_sd[s] * Rimps_sd[s])
        eimp_pr += 0.5*np.sum(np.einsum("psqr,sr->pq", EVs_pr_sd[s], Rimps_sd[s]) * Rimps_sd[s])
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

















