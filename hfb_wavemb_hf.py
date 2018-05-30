import numpy as np
import sys
import itertools

import c1c2
import xform

from uhfb import UHFB
import HfbUtil

#Calculates the energy contribution for a single HFB fragment
#We run the sum only over fragment sites to avoid double counting
#This assumes the fragment came from some HFB state, so the only correlation is from pairing

#Take the fragment trace of a matrix product in fragment-space. This assumes both are hermitian
def fragTrace(m1,m2): 
    n = m1.shape[0]
    assert n % 2 == 0 #should split evenly into fragment and bath space
    nf = n//2
    ri = range(n); rf = range(nf)
    tr = 0.
    for f,j in itertools.product(rf,ri):
        tr += m1[f,j]*m2[j,f]
    return tr

def fragTraceNp(m1,m2):
    n = m1.shape[0]
    assert n % 2 == 0 #should split evenly into fragment and bath space
    nf = n//2
    ri = range(n); rf = range(nf)
    tr = np.trace(np.dot(m1[:,:nf].T,m2[:,:nf]))
#    efrag = np.trace(np.dot(F_f[:,:nf].T,P1_f[:,:nf]))/nf
    return tr

if __name__ == "__main__":
    import pickle

#    file_name = 'bogil_n4u4a4b3.pk'
#    file_name = 'bogil_randHV_n4u4a4b3.pk'
    file_name = 'bogil_coulV_n4u4a4b3.pk'
#    file_name = 'bogil_lrHop_n4u4a4b3.pk'
#    file_name = 'bogil_randV_n4u4a4b3.pk'
#    file_name = 'bogil_rH_n4u4a4b3.pk'

#    file_name = 'bogil_randHV_n4uP2a2b2.pk'

    with open(file_name,'rb') as file_object:
        bogil = pickle.load(file_object)
    
    nf = 1
    n = bogil.n
    VU = UHFB.genVU(bogil.H_app)
    h_dirsum = HfbUtil.matDirSum([bogil.h, bogil.h])

#    print("VU:")
#    print(VU)
#    print("G:")
#    print(bogil.G)
#    print(np.linalg.norm(bogil.G - np.dot(VU,VU.T)))


    #make a list of range(nf) that tile n
    fraglist = []
    for i in range(n//nf): #assumes n divisible by nf
        fraglist.append(range(nf*i,nf*(1+i)))

    e_list1 = []; e_list2 = []; e_list3 = []; e_list4 = []
    for frag_sites in fraglist:
        frag_sites = np.array(frag_sites)

        ###Wavefunction Embedding Process
        Wf = np.zeros((4*nf,2*n)) #2*nf for HF spins a&b + 2*nf for HFB spins a&b. Only 2n because it's only the occupied block
        Wf[:nf,:] = VU[frag_sites,:]
        Wf[nf:2*nf,:] = VU[frag_sites+n,:]
        Wf[2*nf:3*nf,:] = VU[frag_sites+2*n,:]
        Wf[3*nf:,:] = VU[frag_sites+3*n,:]
        u,s,v = np.linalg.svd(Wf)
        C = VU.dot(v.T)

        #print(frag_sites)
        #print(Wf)
        #print(u)
        #print(s)
        #print(v)
        #print("C:")
        #print(C)

        T = np.zeros((4*n,8*nf)) #UHF twice HF and HFB twice HF
        #need to pull out the same set of sites as the SVD
        T[frag_sites,:4*nf] = C[frag_sites,:4*nf] #fill in the fragment V-section
        T[frag_sites+n,:4*nf] = C[frag_sites+n,:4*nf] 
        T[frag_sites+2*n,:4*nf] = C[frag_sites+2*n,:4*nf] 
        T[frag_sites+3*n,:4*nf] = C[frag_sites+3*n,:4*nf] 


        T[:,4*nf:] = C[:,:4*nf] - T[:,:4*nf] #fill in the bath section by subtracting off the fragment



        T_overlap = T.T.dot(T)
        T_norm = T.dot(np.diag(np.diag(T_overlap)**-0.5)) #should it be normalized within pairing blocks?

        T_Vv = T_norm[:2*n,:] #cut T into the original V and U parts
        T_Uv = T_norm[2*n:,:]

        ###Can I normalize it this way instead? --> this is ludicrously wrong
        #T_Vv = T_Vv.dot(np.diag(np.diag(T_Vv.T.dot(T_Vv))**-0.5)) 
        #T_Uv = T_Uv.dot(np.diag(np.diag(T_Uv.T.dot(T_Uv))**-0.5)) 

        #print(T_norm)
        #print()
        #print(T_Vv)
        #print(T_Uv)

#        ##What if I don't normalize it? --> it is very wrong
#        T_Vv = T[:2*n,:] #cut T into the original V and U parts
#        T_Uv = T[2*n:,:]

        F_f = xform.one(T_Vv,h_dirsum+bogil.H[:2*n,:2*n])
        h_f = xform.one(T_Vv,2*h_dirsum)
        P1_f = xform.one(T_Vv,bogil.G[:2*n,:2*n])

        D_f = xform.one(T_Uv, bogil.H[:2*n,2*n:])
        Ki_f = xform.one(T_Uv, bogil.G[:2*n,2*n:])


#        H_f = xform.one(T_norm,bogil.H)
#        G_f = xform.one(T_norm,bogil.G)
#        H_f2 = xform.one(T_norm,c1c2.reduceBath(bogil.H,fb=1.))
#        G_f2 = xform.one(T_norm,c1c2.reduceBath(bogil.G,fb=1.))

        ##This works for hubb and coulV, but not lrHop or randHV
        e_list1.append(2*fragTrace(F_f,P1_f)) #off by precisely a factor of 2. I'll add this factor here for now
        e_list2.append(2*fragTrace(D_f,Ki_f))
        #e_list3.append(2*fragTrace(h_f,P1_f))
        #e_list4.append(e_list1[-1]-e_list3[-1])

        
        e_list3.append(2*np.sum(F_f*P1_f)) #off by precisely a factor of 2. I'll add this factor here for now
        e_list4.append(2*np.sum(D_f*Ki_f))


        ##fragTraceNp appears to get the same energy as fragTrace for randHV & other h,V's
        #e_list3.append(2*fragTraceNp(F_f,P1_f)) #off by precisely a factor of 2. I'll add this factor here for now
        #e_list4.append(2*fragTraceNp(D_f,Ki_f))

##        e_list2.append(np.trace(np.dot(H_f,G_f)))   #wrong
##        e_list3.append(np.trace(np.dot(H_f2,G_f2))) #wrong
#        e_list2.append(fragTrace(H_f,G_f))   #off by 1. exactly for hubb hV? Is this the 1-p term?
#        e_list3.append(fragTrace(H_f2,G_f2)) #wrong


    print("E1 and E2:")
    print(sum(e_list1)/n)
    print(sum(e_list2)/n)

    print("E3 and E4:")
    print(sum(e_list3)/n)
    print(sum(e_list4)/n)

    uhfb_mf_E = 0.5*np.trace(np.dot(bogil.G[:2*n,:2*n],bogil.H[:2*n,:2*n]+h_dirsum))
    uhfb_ph_E = -0.5*np.trace(bogil.G[:2*n,2*n:].dot(bogil.H[:2*n,2*n:]))
    uhfb_mf_1p_E = 0.5*np.trace(np.dot(bogil.G[:2*n,:2*n],2*h_dirsum))
    uhfb_mf_2p_E = uhfb_mf_E - uhfb_mf_1p_E

    print("Embedded HFB energy:", (sum(e_list1)-sum(e_list2))/n)
    print("HFB Energy:", bogil.E)
    print("HFB Mean-Field Component:", uhfb_mf_E)
    print("HFB Particle-Hole Component:", uhfb_ph_E)
    print("HFB 1-particle-MF Component:", uhfb_mf_1p_E)
    print("HFB 2-particle-MF Component:", uhfb_mf_2p_E)


#    print("UHF Energy:", bogil.uhf_e)



##From the HFB Embedding Method Takashi described
#    Sf = Wf.T.dot(Wf)
#    print Sf
#    print np.linalg.norm(Sf-Sf.T)
#    Sf_e, Sf_v = np.linalg.eigh(Sf)
#    Wf_so = Wf.dot(Sf_v)
#    Wenv_so = Wenv.dot(Sf_v)
