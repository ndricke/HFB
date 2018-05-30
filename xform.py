import numpy
import itertools
import logging

kennyloggins = logging.getLogger("xform")
pract0 = 10**-14

def one(T,h):
    return T.T.dot(h).dot(T)

def two(T,V):
    n, ni = T.shape

    Vi = numpy.zeros((ni,ni,ni,ni))
    
    imps = range(ni)

    if type(V) == float or type(V) == int:
#        for i,j,k,l in itertools.product(imps,imps,imps,imps):
#            for mu in range(n):
#                Vi[i,j,k,l] += T[mu,i]*T[mu,j]*T[mu,k]*T[mu,l]*V
        Vi = twoU(T,V)
    elif len(V.shape) == 4:
        Vhalf = numpy.zeros((ni,n,ni,n ))
        for i in range(n):
            for j in range(n):
#                Vhalf[:,:,i,j] = T.T.dot(V[:,:,i,j]).dot(T)
                Vhalf[:,i,:,j] = T.T.dot(V[:,i,:,j]).dot(T)
        for i in imps:
            for j in imps:
#                Vi[i,j,:,:]  = T.T.dot(Vhalf[i,j,:,:]).dot(T)
                Vi[i,:,j,:]  = T.T.dot(Vhalf[i,:,j,:]).dot(T)

    else: 
        kennyloggins.error( "V shape wrong")
        import sys; sys.exit(1)
    return Vi

def twoU(T,U):
    n, ni = T.shape
    Vi = numpy.zeros((ni,ni,ni,ni))
    imps = range(ni)

    for i,j,k,l in itertools.product(imps,imps,imps,imps):
        for mu in range(n):
            Vi[i,j,k,l] += T[mu,i]*T[mu,j]*T[mu,k]*T[mu,l]*U
    return Vi

def rdm(T,psi):
    rho = psi.dot(psi.T)
    return T.T.dot(rho).dot(T)

def one_hub(T):
    n = T.shape[0]
    h = numpy.diag(numpy.ones(n-1),1)
    h[-1,0] = 1
    h += h.T
    h *= -1
    return T.T.dot(h).dot(T)

def two_hub(T):
    n,m = T.shape
    Uimp = numpy.zeros((m,m,m,m))
    #for i in range(m):
    #    Uimp[i,i,i,i] = 1
    for i in range(m):
     for j in range(m):
      for k in range(m):
       for l in range(m):
        for mu in range(n):
            Uimp[i,j,k,l] += T[mu,i]*T[mu,j]*T[mu,k]*T[mu,l]
    return Uimp

def core(T, V):
    n,m = T.shape
    rc = T.dot(T.T)
    if type(V) == float: #if V comes as an int, we want it to throw an error
        hc = numpy.diag([rc[i,i]*V for i in range(n)])
    elif len(V.shape) == 4:
        hc = coreFock(rc,V)
    return hc

def coreFock(rho,V): 
    sz = len(V)
    F = numpy.zeros((sz,sz)) 
    ri = range(sz)
    for mu,nu,la,si in itertools.product(ri,ri,ri,ri): 
        F[mu,nu] += (2*V[mu,si,nu,la] - V[mu,si,la,nu])*rho[la,si] 
    return F 

def MFpdm2(P1):
    n = P1.shape[0]
    P2 = numpy.zeros((n,n,n,n))
    sz = range(n)
    for i,j,k,l in itertools.product(sz,sz,sz,sz):
        P2[i,j,k,l] = P1[k,i]*P1[j,l]
    return P2

def troy2Lit(P2,V):
    n = P2.shape[0]
    P2lit = numpy.zeros((n,n,n,n))
    sz = range(n)
    for i,j,k,l in itertools.product(sz,sz,sz,sz):
        if (2.0*V[i,j,k,l]-V[i,j,l,k]) > pract0:
            P2lit[k,l,i,j] = P2[i,j,k,l]*V[i,j,k,l]/(2*V[i,j,k,l]-V[i,j,l,k])
    return P2lit


def checkPdmSym(P2):
    pract0 = 10**-14
    n = P2.shape[0]
    nz = range(n)
    manchee = 0
    for i,j,k,l in itertools.product(nz,nz,nz,nz):
        if (P2[i,j,k,l] - P2[j,i,l,k]) >= pract0 or \
        P2[i,j,k,l] - P2[k,l,i,j] >= pract0 or \
        P2[i,j,k,l] - P2[l,k,j,i]  >= pract0 or \
        P2[i,j,k,l] - P2[k,j,i,l]  >= pract0 or \
        P2[i,j,k,l] - P2[l,i,j,k]  >= pract0 or \
        P2[i,j,k,l] - P2[i,l,k,j]  >= pract0 or \
        P2[i,j,k,l] - P2[j,k,l,i] >= pract0: 
            print("Symmetry broken!")
            manchee = 1.0
    print("Manchee!: " + str(manchee))
