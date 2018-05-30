
import numpy as np
import logging
from collections import deque

from frankenstein.lattice import LATTICE
from frankenstein.tools.scf_utils import diis_uscf_update
from frankenstein.tools.hfb_utils import get_hfb_PKs, get_hfb_Rs, get_hfb_Ds, \
    get_hfb_Hs, get_hfb_energy, get_hfb_Hmus, get_hfb_nelecs, get_hfb_chempot, \
        solve_hfb_Hs, get_hfb_diis_errs, get_hfb_init_guess


class HFB(object):
    def __init__(s, m, h, V, e_nuc=0., unrestricted=False):
        s.h = h 
        s.V = V 
        s.m = m
        s.nbas = h.shape[0]
        s.e_nuc = e_nuc
        s.unrestricted = unrestricted


    def _do_hfb(s, conv_power=7, max_iter=100, max_diis=10, opt="diis", \
                   mixing_beta=0.3, Rs0=None):

        conv_flag = False
        errs = []
        conv = 10.**(-conv_power)

        # make initial guess
        if Rs0 is None:
            Rs0 = get_hfb_init_guess(s.h, s.V, s.nbas, s.m, s.unrestricted)
        Rs = Rs0
        Hs = get_hfb_Hs(s.h, s.V, Rs)

        # initialize DIIS lists
        s.Hs_list = [deque([], max_diis) for sp in [0, 1]]
        s.errs_list = [deque([], max_diis) for sp in [0, 1]]

        for iteration in range(max_iter):

            if opt == "damping":
                Rs_old = Rs

            es, Ws, mu = solve_hfb_Hs(Hs, s.m)
            Rs, Ps, Ks = get_hfb_Rs(Ws, ret_PKs=True)

            if opt == "damping":
                for sp in [0, 1]:
                    Rs[sp] = Rs_old[sp]*mixing_beta + Rs[sp]*(1.-mixing_beta)

            s.e_hfb, s.e_hf, s.e_pr = get_hfb_energy(h, Hs, Rs, ret_Ecomp=True)
            nelecs = [np.trace(Ps[sp]) for sp in [0,1]]
            assert(np.allclose(nelecs[0], nelecs[1]))

            Hs = get_hfb_Hs(h, V, Rs)
            mu = get_hfb_chempot(Hs, s.m)

            errs, errs_val = get_hfb_diis_errs(Hs, Rs, mu)
            err_val = np.max(errs_val)

            if err_val < conv:
                s.conv_flag = True
                s.iteration = iteration
                break

            if opt == "diis":
                for sp in [0,1]:
                    Hs_list[sp].append(Hs[sp])
                    errs_list[sp].append(errs[sp])
                Hs = diis_uscf_update(Hs_list, errs_list)

        s.Hs, s.Rs, s.mu = Hs, Rs, mu





if __name__ == "__main__":
    import os
    import numpy as np
    from frankenstein import scf
    from frankenstein.tools.lat_utils import get_Hubbard, get_Hubbard_lr


    nbas = 12
    U = -3
    U_lr = U*0.75
    noccs = [2, 3, 5]

    e_hf_Hubbard = [-7.065456, -10.934674, -18.463203]
    e_pr_Hubbard = [-2.556734,  -3.135629,  -3.436093]
    e_hfb_Hubbard = [-9.622190, -14.070303, -21.899296]
    e_hf_lr = [-15.760719, -31.923309, -72.725640]
    e_pr_lr = [-2.147575,  -1.546308,  -1.424380]
    e_hfb_lr = [-17.908294, -33.469617, -74.150020]

    """Tests for HFB on the Hubbard model."""
    for i, nocc in enumerate(noccs):
        pbc = True if nocc%2 else False
        # Hubbard
        h, V = get_Hubbard(nbas, U, pbc)
        mf = HFB(m=nocc, h=h, V=V)
        mf._do_hfb(opt="damping", max_iter=300)
        assert(np.allclose(mf.e_hf, e_hf_Hubbard[i]))
        assert(np.allclose(mf.e_pr, e_pr_Hubbard[i]))
        assert(np.allclose(mf.e_hfb, e_hfb_Hubbard[i]))


        # lr
        h, V = get_Hubbard_lr(nbas, U, U_lr, pbc)
        mf = HFB(m=nocc, h=h, V=V)
        mf._do_hfb(opt="damping", max_iter=300)
        assert(np.allclose(mf.e_hf, e_hf_lr[i]))
        assert(np.allclose(mf.e_pr, e_pr_lr[i]))
        assert(np.allclose(mf.e_hfb, e_hfb_lr[i]))


























