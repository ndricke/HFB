import numpy as np
import itertools
import copy

import embed.schmidt
import est.xform
import est.hf
import embed.dmetenergy

np.set_printoptions(precision=4,suppress=True)

#Numerical recipes implementation of bracket method to find sufficient range for finding a root
def bracket(f,xbrack,maxiter=50,xconst=1.6):
    if xbrack[0] == xbrack[1]: raise ValueError("guess should be 2 different values")
    f0 = f(xbrack[0]); f1 = f(xbrack[1]);
    for i in range(maxiter):
        if f0*f1 < 0: return xbrack,True
        elif abs(f0) < abs(f1):
            xbrack[0] += xconst*(xbrack[0]-xbrack[1])
        else:
            xbrack[1] += xconst*(xbrack[1]-xbrack[0])
    return xbrack,False

def antisymMat(n):
    Z = (np.random.rand(n,n)-0.5)*0.5
    Z = Z+Z.T
    for i in range(n):
        Z[i,i] = 0.
        for j in range(i):
            Z[i,j] *= -1.
    return Z

def unitMatGen(n):
    Z = antisymMat(n)
    U = scl.expm(Z)
    B = 2.*(np.random.rand(n)-0.5)
    B = np.diag(B)
    A = U.dot(B.dot(U.T))
    return Z,U,B,A

def redFock(n,rho,h,V):
    F_red = h.copy()
    ri = range(n)
    for mu,nu,la,si in itertools.product(ri,ri,ri,ri):
        F_red[mu,nu] += (V[mu,si,nu,la] - 0.5*V[mu,si,la,nu])*rho[la,si]
#    F_red = c1c2.hRedHalf
    return F_red

#Input numpy matrix. Reduce f-b terms by 0.5, and zero the b-b terms
def reduceBath(M,fb=0.5,bb=0.):
    n = M.shape[0]
    assert n % 2 == 0
    nf = n//2
    Mred = M.copy()
    Mred *= fb
    Mred[nf:,nf:] *= bb
    Mred[:nf,:nf] = M[:nf,:nf]
    return Mred

#for f,j in itertools.product(niz,nz):
#    frag.e1 += (hf[f,j]-frag.hc[f,j]) * P1f[j,f]

##Fragment 2e Energy
#for f,j,k,l in itertools.product(f_index,niz,niz,niz):
#    e2 += frag.V[f,j,k,l]*P2f[f,j,k,l]
#e2 /= frag.ni*4.

def embed_e1(nf,h,P1):
    n = len(h) 
    ri = range(n) 
    rf = range(nf) 
     
    e1 = 0
    for f,j in itertools.product(rf,ri): 
        e1 += h[f,j]*P1[j,f] 
    return e1

def embed_energy(nf,h,hc,V,P1,P2): 
    n = len(h) 
    ri = range(n) 
    rf = range(nf) 
    print("n, nf: ", n, nf)
    
    h_mc = h - 0.5*hc #we have h+hc here, but for energy evaluation it's only 2h+hc in the end

    e1 = 0; e2 = 0
    for f,j in itertools.product(rf,ri): 
        e1 += 2.*h_mc[f,j]*P1[j,f] 
 
    for f,j,k,l in itertools.product(rf,ri,ri,ri): 
        e2 += V[f,j,k,l]*P2[f,j,k,l] 
#        e2 += (2*V[f,j,k,l]-V[f,j,l,k])*P1[f,k]*P1[j,l] 

    print("e1: ", e1/nf)
    print("e2: ", e2/nf)

    return e1 + e2

def engV(n,nf,V,rho):
    ri = range(n)
    rf = range(nf)
    e2 = 0.
    for f,j,k,l in itertools.product(rf,ri,ri,ri):
        e2 += (2*V[f,j,k,l]-V[f,j,l,k])*rho[f,k]*rho[j,l]
    return e2


def _calc_energy(nf,h,hc,V,P1,P2): #XXX
    e1 = 0.; e2 = 0.; core = 0.
    nz = range(2*nf)
    niz = range(nf) #we always organize fragment, then bath

    for f,j in itertools.product(niz,nz):
        e1 += 2.*(h[f,j]-0.5*hc[f,j]) * P1[j,f]
#        e1 += 2.*h[f,j] * P1[j,f]
#    for f in niz:
#        core += hc[f,f]*P1[f,f]

    #Fragment 2e Energy
    for f,j,k,l in itertools.product(niz,nz,nz,nz):
        e2 += V[f,j,k,l]*P2[f,j,k,l]

#    e = e1 + e2 + core
#    return e, e1, e2, core
    e = e1 + e2
    return e, e2, e2

def genP2(rho):
    n = rho.shape[0]
    ri = range(n) 
    P2 = np.zeros((n,n,n,n))
    for i,j,k,l in itertools.product(ri,ri,ri,ri):
        P2[i,j,k,l] = rho[k,i]*rho[l,j]
    return P2

def genP2f(rho):
    n = rho.shape[0]
    ri = range(n) 
    P2 = np.zeros((n,n,n,n))
    for i,j,k,l in itertools.product(ri,ri,ri,ri):
        P2[i,j,k,l] = 2*rho[i,k]*rho[j,l] - rho[i,l]*rho[j,k]
    return P2

def hf_energy(h,V,rho):
    e1 = np.trace(h.dot(rho)) * 2
    e2 = 0

    n = len(h)
    ri = range(n)
    for i,j,k,l in itertools.product(ri,ri,ri,ri): 
        e2 += (2*V[i,j,k,l]-V[i,j,l,k])*rho[i,k]*rho[j,l] 

    print("e1: ", e1)
    print("e2: ", e2)
    return e1 + e2

def Vhubbard(n, U):
    V = np.zeros((n,n,n,n))
    for i in range(n):
        V[i,i,i,i] = U
    return V

def hRedHalf(nf,h,T):
    h_r = copy.copy(h)
    h_r[nf:,:] = 0.
    return est.xform.one(T,h_r)

def VRedHalf(nf,V,T):
    V_r = copy.copy(V)
    V_r[nf:,:,:,:] = 0.
    return est.xform.two(T,V_r)

def VSymHalf(nf,V,T):
    V_r = copy.copy(V)
    V_r[nf:,:,:,:] *= 0.5
    V_r[:,nf:,:,:] *= 0.5
    V_r[:,:,nf:,:] *= 0.5
    V_r[:,:,:,nf:] *= 0.5
#    V_r[nf:,nf:,:,:] *= 0.5
#    V_r[nf:,:,nf:,:] *= 0.5
#    V_r[nf:,:,:,nf:] *= 0.5
#    V_r[:,nf:,nf:,:] *= 0.5
#    V_r[:,nf:,:,nf:] *= 0.5
#    V_r[:,:,nf:,nf:] *= 0.5
    V_r[nf:,nf:,nf:,nf:] *= 0.0
    return est.xform.two(T,V_r)

def hRedSym(nf,h,T):
    h_r = copy.copy(h)
    h_r[nf:,:nf] *= 0.5
    h_r[:nf,nf:] *= 0.5
    h_r[nf:,nf:] *= 0.0
    return est.xform.one(T,h_r)

def hubb(n,U,pbc=False): #pbc default set to anti-periodic boundary conditions
    V = Vhubbard(n,U)
    h = np.diag(np.ones(n-1),1)
    if pbc==True: h[0,-1] = 1
    elif pbc==False: h[0,-1] = -1
    h += h.T
    h *= -1
    return h,V

def lrHop(n,U,pbc=True,c=0.5,cutoff=6):
    h,V = hubb(n,U)
    for i in range(n):
        for j in range(i):
            dist, pbc_mark = hubRange(i,j,n)
            if i != j and dist <= cutoff and dist != 1:
                if pbc: h[i,j] = -2*pbc_mark*c**dist
                else: h[i,j] = -2*c**dist
                h[j,i] = h[i,j]
    return h,V

#c corresponds to real distance, determines drop-off rate c/r
def coulV(n,U,pbc=False,c=0.5,cutoff=20.0):
    h, V = hubb(n,U,pbc)
    for i in range(n):
        for j in range(i):
            dist,pbc_mark = hubRange(i,j,n)
            if i != j and dist <= cutoff:
                V[i,j,i,j] = U*c/dist
                V[j,i,j,i] = U*c/dist
    return h, V

def randHubb(n,U,pbc=False,scale=0.1):
    h, V = hubb(n,U,pbc)
    for i in range(n):
        V[i,i,i,i] += V[i,i,i,i]*(np.random.rand()-0.5)*scale
    return h, V

def hubRange(i,j,N): 
    dist_naive = np.abs(i-j) 
    if dist_naive > N/2: 
        dist = N - dist_naive 
        pbc_mark = -1 
    else: 
        dist = dist_naive 
        pbc_mark = 1 
    return dist, pbc_mark 

def randHV(n,U=1.,h_scale=0.1,V_scale=0.1):
    h,V = hubb(n,U)
    h_rand = (np.random.rand(n,n)-0.5)*h_scale
    h_rand += h_rand.T
    h += h_rand
    V_rand = np.zeros((n,n,n,n))
    nz = range(n)
    for i,j,k,l in itertools.product(nz,nz,nz,nz):
        V_rand[i,j,k,l] = np.random.rand()*V_scale*U
        V_rand[j,i,l,k] = V_rand[i,j,k,l]
        V_rand[k,l,i,j] = V_rand[i,j,k,l]
        V_rand[l,k,j,i] = V_rand[i,j,k,l]
        V_rand[k,j,i,l] = V_rand[i,j,k,l]
        V_rand[l,i,j,k] = V_rand[i,j,k,l]
        V_rand[i,l,k,j] = V_rand[i,j,k,l]
        V_rand[j,k,l,i] = V_rand[i,j,k,l]
    V += V_rand
    return h,V

def C1Proj(nf, psi, frag_sites, h, V):
    T = embed.schmidt.schmidt(psi,frag_sites,allv=True)
    C1 = T[:,:2*nf]
    Tc = T[:,2*nf:]
    hc = est.xform.core(Tc,V)
#    h_f = xform.one(C1,h)
    h_f = est.xform.one(C1,h+hc)
    V_f = est.xform.two(C1,V)
    return C1, Tc, hc, h_f, V_f

def C1svd(nf, psi, frag_sites, h, V):
    n = h.shape[0]
    M = psi[frag_sites,:] #select the first nf orbitals out of the HF wavefunction
    u,s,v = np.linalg.svd(M) #perform an SVD of the fragment AOs wrt the MOs
    C = psi.dot(v.T) #wavefunction dot SVD
    T = np.zeros((n,2*nf)) #pull components; split entangled bath and fragment into block diagonal matrix
    T[frag_sites,:nf] = C[frag_sites,:nf]
    T[:,nf:] = C[:,:nf] - T[:,:nf]
    Tc = C[:,nf:]
    C1 = copy.copy(T)
    print("Psi:")
    print(psi)
    print("C:")
    print(C)
    print("C1 before:")
    print(C1)
    ##Equivalent formulation for normalizing C1
    #C1_overlap = C1.T.dot(C1)
    #C1_norm = C1.dot(np.diag(np.diag(C1_overlap)**-0.5)) #should it be normalized within pairing blocks?
    for i in range(nf):
        C1[:,nf+i] /= np.sqrt(C1[:,nf+i].dot(C1[:,nf+i]))
        C1[:,i] /= np.sqrt(C1[:,i].dot(C1[:,i]))
    print("C1 after:")
    print(C1)
    hc = est.xform.core(Tc,V)
    h_f = est.xform.one(C1,h+hc)
    V_f = est.xform.two(C1,V)
#    hc_f = est.xform.one(C1,core)
    return C1, Tc, hc, h_f, V_f

#Gswp is the 1-particle density matrix, nf is the number of fragment sites
#note: for HFB, double nf as we have additional non-number converving blocks
def C2swp(Gswp, nf):
    Gimp = Gswp[:nf,:nf] #pull Gimp out of re-organized density matrix
    Gic = Gswp[:,:nf] #pull out [[Gimp],[Gc]]
    eGi,vGi = np.linalg.eigh(Gimp) #diagonalize Gimp so we can get A
    sq_eGi = np.diag(np.sqrt(eGi)) #take sqrt of eigenvalues then turn it into a matrix
    A = np.dot(vGi,sq_eGi) #AA.T = Gimp, and A = U.dot(e^0.5)
    ATinv = np.linalg.inv(A.T)
    C2 = Gic.dot(ATinv)
    return C2

#need to figure out how to pull out the appropriate terms given a list of sites
def C2(G, sites):
#    Gimp = Gswp[:nf,:nf] #pull Gimp out of re-organized density matrix
#    Gic = Gswp[:,:nf] #pull out [[Gimp],[Gc]]
    eGi,vGi = np.linalg.eigh(Gimp) #diagonalize Gimp so we can get A
    sq_eGi = np.diag(np.sqrt(eGi)) #take sqrt of eigenvalues then turn it into a matrix
    A = np.dot(vGi,sq_eGi) #AA.T = Gimp, and A = U.dot(e^0.5)
    ATinv = np.linalg.inv(A.T)
    C2 = Gic.dot(ATinv)

    return C2

if __name__ == "__main__":
    n = 12
    m = 4
    U = 2.
    nf = 2

    h,V = randHV(n,U=U,scale=0.)

    print("Beginning HF")
    hart = hf.HF(h,V,m)
    psi = hart.get_Cocc()
    rho = hart.get_rho()
    print(hart.get_energy()/n)

    #Method 1, producing C1
    M = psi[:nf,:] #select the first nf orbitals out of the HF wavefunction
    S_M = np.dot(M.T,M) #calculate the overlap matrix of the fragment orbitals
    u,s,v = np.linalg.svd(M) #perform an SVD of the fragment overlap matrix
    C = psi.dot(v.T) #wavefunction dot SVD
    T = np.zeros((n,2*nf)) #pull components out of C; split entangled bath and fragment into block diagonal matrix
    T[:nf,:nf] = C[:nf,:nf]
    T[nf:,nf:] = C[nf:,:nf]
    Tc = C[:,nf:]

    C1 = copy.copy(T)
    #C1[:nf,:nf] = np.eye(nf)

    for i in range(nf):
        C1[:,nf+i] /= np.sqrt(C1[:,nf+i].dot(C1[:,nf+i]))
        C1[:,i] /= np.sqrt(C1[:,i].dot(C1[:,i]))

    h_f1 = est.xform.one(C1,h)
    V_f1 = est.xform.two(C1,V)
    core = est.xform.core(Tc,V)
    rho_f1 = est.xform.one(C1,rho+core)
    rho2 = genP2(rho)
    rho_f2 = est.xform.two(C1,rho2)

    print(embed_energy(nf,h_f1,est.xform.one(C1,core),V_f1,rho_f1,rho_f2)/nf)

    #Method 2 for producing C2, which is a unitary rotation of C1
    rho_f = rho[:nf,:nf]
    Gam,U = np.linalg.eigh(rho_f)
    A = U.dot(np.diag(Gam**0.5))
    ATinv = np.linalg.inv(A.T)
    B = rho[nf:,:nf].dot(ATinv)

    rhoh = np.eye(n) - rho
    rhoh_f = rhoh[:nf,:nf]
    Gamh,Uh = np.linalg.eigh(rhoh_f)
    Ah = Uh.dot(np.diag(Gamh**0.5))
    AhTinv = np.linalg.inv(Ah.T)
    Bh = rhoh[nf:,:nf].dot(AhTinv)

    C2 = np.zeros((n,2*nf))
    C2[:nf,:nf] = A
    C2[:nf,nf:] = Ah
    C2[nf:,:nf] = B
    C2[nf:,nf:] = Bh

    h_f2 = est.xform.one(C2,h)
    V_f2 = est.xform.two(C2,V)
    rho_f2 = est.xform.one(C2,rho+core)

    h_r0 = hRedHalf(nf,h,C2)
    V_f2_0 = VRedHalf(nf,V,C2)

    h_r05 = hRedSym(nf,h,C2)

    print((2.*np.trace(h_r0.dot(rho_f2)) + engV(2*nf,2*nf,V_f2_0,rho_f2))/nf)
    print((2.*np.trace(h_r05.dot(rho_f2)) + engV(2*nf,2*nf,V_f2_0,rho_f2))/nf)
















