import numpy as np
import sys

import est.xform
import embed.schmidt
import embed.c1c2

from uhfb import UHFB
import HfbUtil
#import troytest

np.set_printoptions(precision=3,suppress=True)

def rowcolSwp(mtrx,i,j):
    mtrx[:,[i,j]] = mtrx[:,[j,i]]
    mtrx[[i,j]] = mtrx[[j,i]]


##fragment_sites here is only for unique real-space orbitals of fragment
def CollectFrag(mtrx, fragment_sites):
    m_swp = mtrx.copy()
    nf = len(fragment_sites)
    for i,rc in enumerate(fragment_sites): #swap fragment to be in upper-left of p-p and p-h blocks for alpha&beta spins
        rowcolSwp(m_swp, rc,i)
        rowcolSwp(m_swp, rc+1*n, i+1*n)
        rowcolSwp(m_swp, rc+2*n, i+2*n)
        rowcolSwp(m_swp, rc+3*n, i+3*n)
    for i in range(nf): #swap blocks from hole-hole to be next to fragment particle-particle block
        rowcolSwp(m_swp, i+1*nf, i+1*n)
        rowcolSwp(m_swp, i+2*nf, i+2*n)
        rowcolSwp(m_swp, i+3*nf, i+3*n)
    return m_swp

##Rearrange transformation matrix C back to the original configuration before CollectFrag
def RowRestore(C, fragment_sites):
    C_swp = C.copy()
    nf = len(fragment_sites)
    for i in range(nf): #swap blocks from hole-hole to be next to fragment particle-particle block
        C_swp[[i+1*n,i+1*nf]] = C_swp[[i+1*nf,i+1*n]]
        C_swp[[i+2*n,i+2*nf]] = C_swp[[i+2*nf,i+2*n]]
        C_swp[[i+3*n,i+3*nf]] = C_swp[[i+3*nf,i+3*n]]
    for i,rc in enumerate(fragment_sites): #swap fragment to be in upper-left of p-p and p-h blocks for alpha&beta spins
        C_swp[[rc,i]] = C_swp[[i,rc]]
        C_swp[[rc+1*n,i+1*n]] = C_swp[[i+1*n,rc+1*n]]
        C_swp[[rc+2*n,i+2*n]] = C_swp[[i+2*n,rc+2*n]]
        C_swp[[rc+3*n,i+3*n]] = C_swp[[i+3*n,rc+3*n]]
    return C_swp

##Map fragment_sites from real space orbitals to the 4x larger UHFB space
def Real2UhfbFragments(n, fragment_sites):
    nf = len(fragment_sites)
    uhfb_frag_sites = np.zeros(4*nf) #This will store fragment_sites mirrored over each of the 4 types of c/a operators
    fragment_arr = np.array(fragment_sites) #So we can add units of n to it

    uhfb_frag_sites[:nf] = fragment_arr
    uhfb_frag_sites[nf:2*nf] = fragment_arr + n
    uhfb_frag_sites[2*nf:3*nf] = fragment_arr + 2*n
    uhfb_frag_sites[3*nf:] = fragment_arr + 3*n
    uhfb_frag_sites = uhfb_frag_sites.astype(int) #Needs to be ints instead of floats to serve as indices
    return uhfb_frag_sites


##Pass this function a list of fragment sites that spans all 16 UHFB sub-blocks
def CollectFragUHFB(mtrx, fragment_sites):
    ##Fill in fragment block first
    for i,r in enumerate(frag_sites):
        for j,c in enumerate(frag_sites):
            m_swp[i,j] = mtrx[r,c]

    sites = np.arange(n4)
    env_sites = [item for item in sites if item not in frag_sites]
    for i,r in enumerate(env_sites):
        for j,c in enumerate(env_sites):
            m_swp[i+4*nf,j+4*nf] = mtrx[r,c]
    m_swp[:4*nf,4*nf:] = mtrx[:4*nf,env_sites]
    m_swp[4*nf:,:4*nf] = m_swp[:4*nf,4*nf:].T #fill in the symmetric part of the matrix

    return m_swp


##Pass this function a list of fragment sites that spans all 16 UHFB sub-blocks
def ReduceBathUHFB(mtrx, uhfb_fragment_sites):
    m_red = mtrx.copy()
    n4 = mtrx.shape[0]
    for i in range(n4):
        for j in range(n4):
            if (i and j) not in uhfb_fragment_sites:
                m_red[i,j] *= 0.
            elif (i or j) not in uhfb_fragment_sites:
                m_red[i,j] *= 0.5
            else:
                pass
    return m_red


if __name__ == "__main__":
    import pickle
#    n, h, V, A, U, D, B = loadTFCI(nf)
#    G = GfromZ(D)

    file_name = 'bogil_n4u4a4b3.pk'
#    file_name = 'bogil_coulV_n4u4a4b3.pk'
#    file_name = 'bogil_lrHop_n4u4a4b3.pk'
#    file_name = 'bogil_randV_n4u4a4b3.pk'
#    file_name = 'bogil_randHV_n4u2a4b4.pk'
#    file_name = 'bogil_randHV_n4uP2a2b2.pk'
#    file_name = 'bogil_n4uP2a4b4.pk'

    with open(file_name,'rb') as file_object:
        bogil = pickle.load(file_object)

    nf = 1
    n = bogil.n
    VU = UHFB.genVU(bogil.H_app)
    h_dirsum = HfbUtil.matDirSum([bogil.h, bogil.h])
    h_uhfb = HfbUtil.matDirSum([h_dirsum, -1*h_dirsum])

    tol0 = 10**-14 #if it's an HF solution, sqrt function in Schmidt decomp for C2 creates nan if it's negative 0 from numerical error

    #make a list of range(nf) that tile n
    fraglist = []
    for i in range(n//nf): #assumes n divisible by nf
        fraglist.append(range(nf*i,nf*(1+i)))

    e_list = []
    for frag_sites in fraglist:
        uhfb_frag_sites = Real2UhfbFragments(n, frag_sites)
        Hred = ReduceBathUHFB(bogil.H, uhfb_frag_sites)
        hred = ReduceBathUHFB(h_uhfb, uhfb_frag_sites)

        Gswp = CollectFrag(bogil.G, frag_sites)
        Hswp = CollectFrag(bogil.H, frag_sites)
        hswp = CollectFrag(h_uhfb, frag_sites)
        H_app_swp = CollectFrag(bogil.H_app, frag_sites)
        H_red_swp = CollectFrag(Hred, frag_sites)
        h_red_swp = CollectFrag(hred, frag_sites)

        #Gswp_back = CollectFrag(Gswp, frag_sites) #Applying CollectFrag twice does successfuly reproduce an identity operation


#        Gswp = CollectFragUHFB(bogil.G, frag_sites) #Doesn't pass GH-HG test after swapping
#        H_app_swp = CollectFragUHFB(bogil.H_app, frag_sites)
#        GH = Gswp.dot(H_app_swp)-H_app_swp.dot(Gswp) #A+
#        print(np.linalg.norm(GH))
#        print(bogil.G.dot(bogil.H_app)-bogil.H_app.dot(bogil.G)) #A+

        Gimp = Gswp[:4*nf,:4*nf] #pull Gimp out of re-organized density matrix
        Gic = Gswp[:,:4*nf] #pull out [[Gimp],[Gc]]

        eGi,vGi = np.linalg.eigh(Gimp) #diagonalize Gimp so we can get A
        #eGi[np.abs(eGi) < tol0] = 0 #only necessary for HF solutions
        sq_eGi = np.diag(np.sqrt(eGi)) #take sqrt of eigenvalues then turn it into a matrix
        A = np.dot(vGi,sq_eGi) #AA.T = Gimp, and A = U.dot(e^0.5)
        ATinv = np.linalg.inv(A.T)
        C2 = Gic.dot(ATinv)

        ##Rearrange C2 back into the original basis set ordering
        C2 = RowRestore(C2, frag_sites)
        C2_U = C2[:2*n,:]
        C2_V = C2[2*n:,:]

        G_f = C2.T.dot(Gswp).dot(C2) #Why is this the identity matrix? (at least for half-filling)

##I'm not sure I can split the projection in this manner, as I've rearranged the basis from p-p/p-h structure
        P_f = est.xform.one(C2_U, bogil.G[:2*n,:2*n])
        K_f = est.xform.one(C2_V, bogil.G[:2*n,2*n:])
        ##F_f = est.xform.one(C2_U, bogil.H[:2*n,:2*n])
        ##D_f = est.xform.one(C2_V, bogil.H[:2*n,2*n:])
        #F_f = est.xform.one(C2_U, h_dirsum+Hred[:2*n,:2*n])
        F_f = est.xform.one(C2_U, hred[:2*n,:2*n]+Hred[:2*n,:2*n])
        D_f = est.xform.one(C2_V, Hred[:2*n,2*n:])

        efrag = 0.5*np.trace(np.dot(F_f.T,P_f) + np.dot(K_f.T,D_f))
        print(np.trace(np.dot(K_f.T,D_f)))
        


        ##I'm going to reduce these instead before projecting
#        h_f = est.xform.one(C2, hswp)
#        H_f = est.xform.one(C2, Hswp)

        hred_f = est.xform.one(C2, h_red_swp)
        Hred_f = est.xform.one(C2, H_red_swp)

        #efrag = np.trace(G_f.dot(hred_f+Hred_f)) #Doesn't currently work even for testing HF cases
        e_list.append(efrag)
        print("Efrag %s: %s" % (" ".join([str(item) for item in frag_sites]),str(efrag)))

#        V_f = np.zeros((2*nf,2*nf,2*nf,2*nf))
#        print(h_f.shape)
#        print(V_f.shape)
#        fci = troytest.TroyFCI(usetmp=False)
#        hred = fci.calcHred(h_f,V_f)
#        print(hred)
#        e,v = np.linalg.eigh(hred)
#        print(e)
#        print(v)
#
#
    e_arr = np.array(e_list)
    print(np.sum(e_arr))

    print("UHFB Total Energy:", bogil.E)









"""
H = np.zeros((2*n,2*n))
H[:n,:n] = F
H[n:,n:] = -F
H[:n,n:] = Delta 
H[n:,:n] = -Delta
e,W = np.linalg.eigh(H) #This has all the right eigenvalues, but they're mixed with the positive ones

Wocc = W[:,:n] #If I choose W[:,n:], then I'll get a positive energy for the trace of G and H
#G = Wocc.dot(Wocc.T) #sorry bud, we're going to make sure we can recover the right energy w/o you atm
G = np.zeros((2*n,2*n))
G[:n,:n] = rho
G[n:,n:] = np.eye(n) - rho
"""
