"""
A script doing HF-in-HF embedding.
"""


import numpy as np
from frankenstein.tools.io_utils import parse_xyz, get_ao_ints, dumpMat, dumpVec
from frankenstein.tools.sd_utils import xform_2, xform_4, xform_core
from frankenstein import lattice, scf, be


def hf_in_hf_emb(mf, fragsites):
    mb = be.BE(mf, fragsites=fragsites)
    T = mb.T
    TE = mb.TE
    hs0 = xform_2(mf.h, T)
    hcore = xform_core(mf.V, TE)
    hs_core = xform_2(hcore, T)
    Vs = xform_4(mf.V, T)
    Ps = T.T @ mf.rdm1 @ T
    PE = TE @ TE.T

    Eimp = np.sum((2*hs0 + 2*np.einsum("pqrs,rs->pq", Vs, Ps) - \
        np.einsum("psrq,rs->pq", Vs, Ps)) * Ps)
    Eimpbath = 2 * np.sum(hs_core*Ps)
    Ebath = np.sum((2*mf.h + hcore)*PE)

    return Eimp, Eimpbath, Ebath


if __name__ == "__main__":
    h, V, e_nuc = get_ao_ints(parse_xyz("geom/ch4.zmat"), "sto-3g", True)
    nbas = h.shape[0]
    nocc = 5

    lat = lattice.LATTICE(h=h, V=V, e_nuc=e_nuc, nocc=nocc)
    mf = scf.RHF(lat)
    mf.kernel(verbose="mute")

    # one-frag embedding
    fragsites = [0, 2]
    Eimp, Eimpbath, Ebath = hf_in_hf_emb(mf, fragsites)

    print("Ehf      = % .10f" % mf.e_scf)
    print("Eimp     = % .10f" % Eimp)
    print("Eimpbath = % .10f" % Eimpbath)
    print("Ebath    = % .10f" % Ebath)
    print("sum      = % .10f" % (Eimp+Eimpbath+Ebath))

    print()

    # partition
    fraglist = [[0, 2, 3], [1, 5], [4], [6, 8], [7]]
    Efrag = []
    for fragsites in fraglist:
        Eimp, Eimpbath, Ebath = hf_in_hf_emb(mf, fragsites)
        Efrag.append(Eimp + 0.5*Eimpbath)
    dumpVec(Efrag, operation="sum")
