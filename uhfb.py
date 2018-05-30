import numpy as np
import itertools
import scipy.linalg as scl
import scipy.optimize as sco
import scipy as sc

import HfbUtil

from frankenstein import scf
from frankenstein.tools.scf_utils import get_rdm1, get_uscf_fock, \
     get_uscf_energy, oda_uscf_update, get_uscf_err, diis_uscf_update
from frankenstein.tools.fci_utils import get_abstr
from frankenstein.tools.scf_utils import get_rdm1, get_uscf_fock
    
np.set_printoptions(precision=4,suppress=True)

"""
To convert over to unrestricted UHFB:
1. anti-symmetrized 2ei for generating both Delta and Fock matrices
2. modify UHF code to generate V in unrestricted basis
3. Verify UHF code in unrestricted basis actually works
4. Do I need to change the way mu works to conserve initial alpha and beta electrons separately?
"""


class UHFB(object):
    def __init__(s, m_a, m_b, h, V, rhos, Ki, e_nuc=0., tol=10**-7):
        s.h = h 
        s.V = V 
        s.h_uhf = HfbUtil.matDirSum([s.h,s.h])
        s.m_a = m_a
        s.m_b = m_b
        s.m = m_a + m_b
        s.n = h.shape[0]
        s.tol = tol
        s.app_pot = 0.
        s.mu = 0. #each time we re-calc applied potentials, we'll keep track of the total with mu

        s.rhos = rhos
        s.Ki = Ki

        rho_block = HfbUtil.matDirSum(s.rhos)
        s.G = HfbUtil.assembleG(rho_block,s.Ki)

        s.Fs = get_uscf_fock(s.h,s.V,s.rhos)
        s.Delta = s.genDelta(s.Ki,s.V)
        s.H = s.genH(s.Fs,s.Delta)
        s.E = np.trace(np.dot(s.G[:2*s.n,:2*s.n],s.H[:2*s.n,:2*s.n]+s.h_uhf)-s.G[:2*s.n,2*s.n:].dot(s.H[:2*s.n,2*s.n:]))*0.5
        print("Initial Energy:")
        print(s.E)

    def occDif(s,pot):
        H_app = s.appH(s.H,pot)
        G = s.genG(H_app)
        return np.trace(G[:2*s.n,:2*s.n]) - s.m

    ##Apply a potential to both alpha and beta sub-blocks to tune separately to prevent a drift of m_s
    def occDifAB(s,pot):
        H_app = s.appH(s.H,pot)
        G = s.genG(H_app)
        diff_a = np.trace(G[:s.n,:s.n]) - s.m_a
        diff_b = np.trace(G[s.n:2*s.n, s.n:2*s.n]) - s.m_b
        return (diff_a, diff_b)

#Figure out two potentials that bracket the potential that must be applied for desired population
    def occBrack(s,maxiter=50,xfactor=1.4):
        #calculate whether population is above or below desired quantity with no applied potential
        Umax = 6. #the applied potentials are *probably* not bigger than 8; will still check
        err0 = s.occDif(0.)
        #set one end of bracket to 0
        if err0 < 0: #if the population is negative, we want to put a negative mu to draw particles
            static_brack = 0.
            dyn_brack = -Umax
        elif err0 > 0:
            static_brack = Umax
            dyn_brack = 0.
        elif err0 == 0: return 0.,True
        #iteratively increase other end of bracket until sign changes
        for i in range(maxiter):
            if s.occDif(static_brack)*s.occDif(dyn_brack) < 0:
                #print("occBrack potential range:",sorted([static_brack,dyn_brack]))
                return sorted([static_brack,dyn_brack]),True
            else:
                static_brack = 1.0*dyn_brack
                dyn_brack *= xfactor
        print("occBrack: Failed to find suitable root brackets")
        return None,False

    def occOpt(s):
        brack,success = s.occBrack() #applies nec. mu to tune pop to m
        if success:
            if brack == 0: s.app_pot,conv = (0.,'Zero-Limit')
            else: s.app_pot,conv = sco.brentq(s.occDif,brack[0],brack[1],full_output=True) #XXX unreliable
        else: raise ValueError("Failed to find brackets for root")
        s.mu += s.app_pot
        H_app = UHFB.appH(s.H,s.app_pot)
        G_new = UHFB.genG(H_app)
        return H_app, G_new

    @staticmethod
    def genDelta(Ki,V):
        n = len(V)
        D = np.zeros((2*n,2*n))
        ri = range(n)
        for i,j,k,l in itertools.product(ri,ri,ri,ri):
            D[i,j] += 0.5*V[i,j,k,l]*Ki[k,l]
            D[i,j+n] += 0.5*V[i,j,k,l]*Ki[k,l+n]
            D[i+n,j] += 0.5*V[i,j,k,l]*Ki[k+n,l]
            D[i+n,j+n] += 0.5*V[i,j,k,l]*Ki[k+n,l+n]
#        D[n:,:n] = -1*D[:n,n:]
        return D

#  D_ij = <ij|V|kl>K_kl
#  h_ij = <ik|V|jl>P_lk

#   for mu,nu,la,si in itertools.product(ri,ri,ri,ri):
#       F[mu,nu] += (2*V[mu,si,nu,la] - V[mu,si,la,nu])*rho[la,si]
#       F[i,j] += V[i,k,j,l]*rho[l,k]

    @staticmethod
    def appH(H,pot):
        H_app = H.copy()
        n = H_app.shape[0]//2
        app = np.eye(n)*pot
        H_app[:n,:n] += app
        H_app[n:,n:] -= app
        return H_app

    @staticmethod
    def genH(Fs,Delta):
        n = Fs[0].shape[0]
        H = np.zeros((4*n,4*n))
        H[:n,:n] = Fs[0]
        H[n:2*n,n:2*n] = Fs[1]
        H[2*n:,2*n:] = -1.*H[:2*n,:2*n]
        H[:2*n,2*n:] = Delta 
        H[2*n:,:2*n] = -Delta
        return H

    @staticmethod
    def genVU(H):
        n = int(H.shape[0]//2)
        e,W = np.linalg.eigh(H) #This has all the right eigenvalues, but they're mixed with the positive ones
        #I think I'm actually supposed to be choosing W[:,n:], but I guess the energy comes out wrong?
        VU = W[:,:n] #If I choose W[:,n:], then I'll get a positive energy for the trace of G and H
        return VU

    @staticmethod
    def genG(H):
        n = int(H.shape[0]//2)
        e,W = np.linalg.eigh(H) #This has all the right eigenvalues, but they're mixed with the positive ones
        #I think I'm actually supposed to be choosing W[:,n:], but I guess the energy comes out wrong?
        Wocc = W[:,:n] #If I choose W[:,n:], then I'll get a positive energy for the trace of G and H
#        Wocc = UHFB.genVU(H)
        G = Wocc.dot(Wocc.T) 
        return G

    @staticmethod
    def genWocc(H):
        n = int(H.shape[0]//2)
        e,W = np.linalg.eigh(H) #This has all the right eigenvalues, but they're mixed with the positive ones
        Wocc = W[:,:n] #If I choose W[:,n:], then I'll get a positive energy for the trace of G and H
        return Wocc

    def _do_hfb(s):
        converged = False
        count = 0;
        errs = []

        while not converged:
            count += 1
            if count % 10 == 0: 
                print("HFB Iteration ",count)

            ##determine potentials that maintain correct population
            brack,success = s.occBrack() #applies nec. mu to tune pop to m
            if success:
                if brack == 0: s.app_pot,conv = (0.,'Zero-Limit')
                else: s.app_pot,conv = sco.brentq(s.occDif,brack[0],brack[1],full_output=True) #XXX unreliable
            else: raise ValueError("Failed to find brackets for root")

            s.mu += s.app_pot
            H_app = UHFB.appH(s.H,s.app_pot) #apply potentials to maintain correct population
            G_new = UHFB.genG(H_app) #find eigenvectors of H, and use to create new density matrix

            rhos_G = []
            rhos_G.append(G_new[:s.n,:s.n]) #get the alpha-spin portion of density matrix
            rhos_G.append(G_new[s.n:2*s.n,s.n:2*s.n]) #bet beta-spin
            s.Fs = get_uscf_fock(s.h,s.V,rhos_G) #form unrestricted fock matrix from new density matrix
            s.Delta = s.genDelta(G_new[:2*s.n,2*s.n:],s.V) #form pairing matrix
            s.H = s.genH(s.Fs,s.Delta) #slap the block matrix together

            err = G_new - s.G #stop when the density matrix stops changing
            #err = H_new.dot(G_new) - G_new.dot(H_new) #an alternate convergence criteria
            s.G = G_new

            converged = np.linalg.norm(err) < s.tol*s.n

        print("HFB supposedly converged?")
#        s.E = np.trace(np.dot(s.G[:n,:n],s.H[:n,:n]+s.h-0.*s.mu*np.eye(s.n))-s.G[:n,n:].dot(s.H[:s.n,s.n:]))/s.n
        s.E = np.trace(np.dot(s.G[:2*s.n,:2*s.n],s.H[:2*s.n,:2*s.n]+s.h_uhf)-s.G[:2*s.n,2*s.n:].dot(s.H[:2*s.n,2*s.n:]))*0.5
        s.E_HF = np.trace(np.dot(s.G[:2*s.n,:2*s.n],s.H[:2*s.n,:2*s.n]+s.h_uhf))*0.5
        s.E_pp = np.trace(s.G[:2*s.n,2*s.n:].dot(s.H[:2*s.n,2*s.n:]))*0.5
        F_mod = s.H[:s.n,:s.n]-s.app_pot*np.eye(s.n)
        s.H_app = H_app
        s.eH_app, s.W = np.linalg.eigh(H_app)


if __name__ == "__main__":
    import embed.c1c2 
    import pickle

    U = 2

#    n = 2
#    m_a = 2
#    m_b = 1
#    rho_a = np.array([[0.51,0.],[0.,0.49]])
#    Ki = np.array([[0,0,0.51,0.],[0,0,0.,0.49],[-0.51,0,0.,0.],[0,-0.49,0.,0]])

    n = 4
    m_a = 4
    m_b = 4
#    rho_a = np.array([[0.51,0.,0.,0.],[0.,0.49,0.,0.],[0.51,0.,0.,0.],[0.,0.49,0.,0.]])
##    Ki = np.array([[0,0,0.51,0.],[0,0,0.,0.49],[-0.51,0,0.,0.],[0,-0.49,0.,0]])


    bsym_block = np.diag([0.6,0.4]*(n//2)) #broken symmetry guess used to generate both particle and hole components of density matrix
    rho_a = bsym_block
    Ki = np.zeros((2*n,2*n))
    Ki[:n,n:] = bsym_block
    Ki[n:,:n] = -bsym_block


    rho_guess = [rho_a,rho_a]
    h,V = embed.c1c2.hubb(n,U=U)
    #h,V = embed.c1c2.randHV(n,U=U)
    #h,V = embed.c1c2.randHV(n,U=U,h_scale=0.0)
    #h,V = embed.c1c2.coulV(n,U=U)
    #h,V = embed.c1c2.lrHop(n,U=U)
    #h,V = embed.c1c2.randHubb(n,U=U)

    print(Ki)
    print(h)

    bogil = UHFB(m_a, m_b, h, V, rhos=rho_guess, Ki=Ki)
    bogil._do_hfb()


    print("HFB Converged Energy:",bogil.E)
    print(bogil.H[:2*n,:2*n])
    print(bogil.H[:2*n,2*n:])
    print(bogil.G[:2*n,:2*n])
    print(bogil.G[:2*n,2*n:])
    print()
    print("HFB Energy:", bogil.E)
#    print(bogil.W)
    Wocc = bogil.W[:,:2*n]
#    print(np.dot(Wocc,Wocc.T))
    print(np.linalg.norm(np.dot(bogil.H_app,bogil.G)-np.dot(bogil.G,bogil.H_app)))
    print(np.linalg.norm(bogil.G - np.dot(bogil.G,bogil.G)))

    umf = scf.UHF(h=h, V=V, e_nuc = 0, nocc=m_a, noccb=m_b, opt='diis')
    umf.kernel(D0=rho_guess)
    print(umf.e_tot)
    print(umf.rdm1)
    bogil.uhf_e = umf.e_tot
    bogil.uhf_rdm1 = umf.rdm1

#    mf = scf.RHF(h=h, V=V, enuc=0, nocc=1)
#    mf.kernel(D0=rho0)
#    print(mf.e_tot)
#    print(mf.rdm1)


    file_name = 'bogil_n4uP2a4b4.pk'
    with open(file_name,'wb') as file_object:
        pickle.dump(bogil, file_object)


























