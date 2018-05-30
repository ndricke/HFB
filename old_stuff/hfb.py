import numpy as np
import itertools
import scipy.linalg as scl
import scipy.optimize as sco
import scipy as sc

import sys

import est.xform
import embed.c1c2 
import est.hf #def formH(P,K,h,V):
    
np.set_printoptions(precision=4,suppress=True)

class HFB(object):
    def __init__(s,h,V,m,rho=None,Ki=None,tol=10**-7):
        s.h = h
        s.V = V
        s.m = m
        s.n = h.shape[0]
        s.tol = tol
        s.app_pot = 0.
        s.mu = 0. #each time we re-calc applied potentials, we'll keep track of the total with mu

        s.rho = rho
        s.Ki = Ki
        
        s.G = np.zeros((2*s.n,2*s.n))
        s.G[:s.n,:s.n] = s.rho
        s.G[s.n:,s.n:] = np.eye(s.n)-s.rho
        s.G[:s.n,s.n:] = s.Ki
        s.G[s.n:,:s.n] = -1.*s.Ki

        s.F = scf.RHF.get_fock(s.h,s.V,s.rho)
        s.Delta = s.genDelta(s.Ki,s.V)
        s.H = s.genH(s.F,s.Delta)


    def occDif(s,pot):
        H_app = s.appH(s.H,pot)
        G = s.genG(H_app)
        return np.trace(G[:s.n,:s.n]) - s.m

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
                print("occBrack potential range:",sorted([static_brack,dyn_brack]))
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
        H_app = HFB.appH(s.H,s.app_pot)
        G_new = HFB.genG(H_app)
        return H_app, G_new

    def _do_hfb(s, iter_max=300):
        converged = False
        errs = []
        Hs = []
        for i in range(iter_max):
            if i % 10 == 0: print("HFB Iteration ",i)
            H_app, G_new = s.occOpt()

            #brack,success = s.occBrack() #applies nec. mu to tune pop to m
            #if success:
            #    if brack == 0: s.app_pot,conv = (0.,'Zero-Limit')
            #    else: s.app_pot,conv = sco.brentq(s.occDif,brack[0],brack[1],full_output=True) #XXX unreliable
            #else: raise ValueError("Failed to find brackets for root")
            #s.mu += s.app_pot
            #H_app = HFB.appH(s.H,s.app_pot)
            #G_new = HFB.genG(H_app)
            #print("trace(G[:n,:n]) - target: ", np.trace(G_new[:s.n,:s.n])-s.m)

            s.F = scf.RHF.get_fock(s.h, s.V, G_new[:s.n,:s.n]) #is HFB the same as HF for this part?
            s.Delta = HFB.genDelta(G_new[:s.n,s.n:],s.V) #are we applying this correctly?
            s.H = HFB.genH(s.F,s.Delta)

            #err = H_new.dot(G_new) - G_new.dot(H_new)
            err = G_new - s.G
            s.G = G_new

            converged = np.linalg.norm(err) < s.tol*s.n
            if converged: break

        print("HFB supposedly converged?")
#        s.E = np.trace(np.dot(s.G[:n,:n],s.H[:n,:n]+s.h-0.*s.mu*np.eye(s.n))-s.G[:n,n:].dot(s.H[:s.n,s.n:]))/s.n
        s.E = np.trace(np.dot(s.G[:s.n,:s.n],s.F[:s.n,:s.n]+s.h)-s.G[:s.n,s.n:].dot(s.H[:s.n,s.n:]))/s.n
        F_mod = s.H[:s.n,:s.n]-s.app_pot*np.eye(s.n)
        s.H_app = H_app
        s.eH_app, s.W = np.linalg.eigh(H_app)

    @staticmethod
    def calcEnergy(G,h,H):
        n = G.shape[0]//2
        e_fock = np.trace(np.dot(G[:n,:n],H[:n,:n]+h))/n
        e_DKi = np.trace(np.dot(G[:n,n:],H[:n,n:]))/n
#        return np.trace(np.dot(G[:n,:n],H[:n,:n]+h)-G[:n,n:].dot(H[:n,n:]))/n
        return (e_fock, e_DKi)

    @staticmethod
    def genDelta(ki,V):
        n = len(V)
        D = np.zeros((n,n))
        ri = range(n)
        for i,j,k,l in itertools.product(ri,ri,ri,ri):
            D[i,j] += (V[i,j,k,l]-0.5*V[i,j,l,k])*ki[k,l]
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
    def genH(F,Delta):
        n = F.shape[0]
        H = np.zeros((2*n,2*n))
        H[:n,:n] = F
        H[n:,n:] = -F
        H[:n,n:] = Delta 
        H[n:,:n] = -Delta
        return H

    @staticmethod
    def genG(H):
        n = int(H.shape[0]//2)
        e,W = np.linalg.eigh(H) #This has all the right eigenvalues, but they're mixed with the positive ones
        #I think I'm actually supposed to be choosing W[:,n:], but I guess the energy comes out wrong?
        Wocc = W[:,:n] #If I choose W[:,n:], then I'll get a positive energy for the trace of G and H
        G = Wocc.dot(Wocc.T) 
        return G

    @staticmethod
    def genWocc(H):
        n = int(H.shape[0]//2)
        e,W = np.linalg.eigh(H) #This has all the right eigenvalues, but they're mixed with the positive ones
        Wocc = W[:,:n] #If I choose W[:,n:], then I'll get a positive energy for the trace of G and H
        return Wocc

if __name__ == "__main__":
    import frankenstein.tools.io_utils as iou
    from frankenstein import scf
    from frankenstein import lattice, scf

    n = 2
    U = -2.
    m = 1

#    nf = 4
#    cl_list = []

#    h,V = embed.c1c2.randHV(n,U=U,scale=0.1)
    h,V = embed.c1c2.hubb(n,U=U)
    print(h)
    h = np.array([[0,-1],[-1,0]])
    print(h)
    print(V)
#    D_rnsym = c1c2.antisymMat(n)*0.

#    rho_guess = np.array([[0.5,0.0],[0.0,0.5]])
#    Ki_guess = np.array([[0.5,0.5],[-0.5,0.5]])

    rho_guess = np.array([[0.5,0.25],[0.25,0.5]])
    Ki_guess = np.array([[0.,0.25],[-0.25,0.]])

#    print(rho_guess)
#    print(Ki_guess)
    

    bogil = HFB(h,V,m,Ki=Ki_guess,rho=rho_guess)
    print("G_before")
    print(bogil.G)
    bogil._do_hfb(1)
    print("G")
    print(bogil.G)
    print("H")
    print(bogil.H)
    E = bogil.calcEnergy(bogil.G,h,bogil.H)
    print(E)


    lat = lattice.LATTICE(h=h,V=V,e_nuc=0, nocc=1)
    mf = scf.RHF(lat)
    mf.kernel()
    print(mf.e_tot)
    print(mf.fock)
    print(mf.rdm1)
    print(mf.mo_coeff)
    
#    umf = scf.UHF(h=h, V=V, e_nuc =0, nocc=1, noccb=1, opt='diis')
#    umf.kernel()
#    print(umf.e_tot)
#    print(umf.fock)
#    print(umf.rdm1)
#    print(umf.mo_coeff)


#    bogil._do_hfb()

#    cl_list.append(bogil)
#    bogil2 = HFB(h,V,m,Delta=D_rnsym)
#    bogil2._do_hfb()
#    cl_list.append(bogil2)
#
#    for inst in cl_list:
##        print "H(Delta): "
##        print inst.H[:n,n:]
#        print("Trace HF rho: ", np.trace(inst.rho))
#        print("Trace HFB G[:n,:n]: ",np.trace(inst.G[:inst.n,:inst.n]))
#        print("E: ", inst.E)
#        print("Fock Energy: ",np.trace(np.dot(inst.G[:inst.n,:inst.n],inst.F+inst.h))/inst.n) #misses non-0 Delta
#        print("Pairing Energy: ",np.trace(np.dot(inst.G[:inst.n,inst.n:],inst.Delta))/inst.n) #misses non-0 Delta
#    print("Diff of F's: ", np.linalg.norm(bogil.F-bogil2.F))
























