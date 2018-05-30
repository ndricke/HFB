import numpy as np
import itertools
import logging

def directSum(M1,M2):
    n1 = M1.shape
    n2 = M2.shape
    M_ds = np.zeros((n1[0]+n2[0],n1[1]+n2[1]))
    M_ds[:n1[0],:n1[1]] = M1
    M_ds[n1[0]:,n1[1]:] = M2
    return M_ds

class UHF:
    def __init__(s,h,V,m_a,m_b,diis=True,tol=1e-8,rho_guess=None):
        s.h = h #in spatial orbital basis
        s.V = V #in spatial orbital basis
        s.m_a = m_a
        s.m_b = m_b
        s.diis = diis
        s.tol = tol

        s.n = s.h.shape[0] #Num rows is the dimension of the density matrix to be
        if type(rho_guess) == np.ndarray:
            s.rho_a = rho_guess[0]
            s.rho_b = rho_guess[1]
            if s.rho.shape[0] != s.n: raise ValueError("Input HF guess has wrong shape")
        else:
            s.rho_a = np.eye(s.n)
            s.rho_b = np.eye(s.n)

        s.rho = directSum(s.rho_a,s.rho_b)
        s.C = None
        s.e = None

    @staticmethod
    def genFock(rho,h,V):
        n = h.shape[0]
        F = directSum(h,h)
        ri = range(len(h))
        for i,j,k,l in itertools.product(ri,ri,ri,ri):
            F[i,j] += (V[i,l,j,k] - V[i,l,k,j])*rho[k,l] + V[i,l,j,k]*rho[k+n,l+n]
            F[i+n,j+n] += (V[i,l,j,k] - V[i,l,k,j])*rho[k+n,l+n] + V[i,l,j,k]*rho[k,l]
        return F

    @staticmethod
    def checkHomoLumoDegen(w,m):
            homo = w[m-1]
            lumo = w[m]
            if abs(lumo-homo) < 1e-4:
                print("DANGER ZONE: HOMO and LUMO are degenerate")

    @staticmethod
    def genRho(F,m):
            w,C = np.linalg.eigh(F)
            UHF.checkHomoLumoDegen(w,m)
            Cocc = C[:,:m]
            rho = Cocc.dot(Cocc.T)
            return rho

    def _do_scf(s):
        F = s.genFock(s.rho,s.h,s.V)

        converged = False
        s.count = 0
        while not converged:
            s.count += 1; print(s.count);
            rho_a_new = s.genRho(F[:s.n,:s.n], s.m_a) #XXX modify these
            rho_b_new = s.genRho(F[s.n:,s.n:], s.m_b)
            rho_new = directSum(rho_a_new, rho_b_new)
            F_new = s.genFock(rho_new,s.h,s.V)
            err = F_new.dot(rho_new) - rho_new.dot(F_new) 

            F = F_new
            converged = np.linalg.norm(err)<s.tol*s.n

        s.rho = rho_new
        s.F = F

    @staticmethod
    def _next_diis(errs, Fs):
        n = len(errs)
        B = np.zeros((n,n))
        for i,j in itertools.product(range(n), range(n)):
            B[i,j] = np.dot(errs[i].ravel(), errs[j].ravel())
        A = np.zeros((n+1, n+1))
        A[:n,:n] = B
        A[n,:] = -1
        A[:,n] = -1
        A[n,n] = 0
        b = np.zeros(n+1)
        b[n] = -1
        try:
            x = np.linalg.solve(A,b)
        except (np.linalg.linalg.LinAlgError):
            print("lin solver fails! Using pinv...")
            P = np.linalg.pinv(A)
            x = P.dot(b)
        w = x[:n]

        F = np.zeros(Fs[0].shape)
        for i in range(n):
            F += w[i] * Fs[i]
        return F

if __name__ == "__main__":
    import est.hf
    np.set_printoptions(precision=3, suppress = True)

    #Hubbard ABAB model
    n = 4
    ma = 2
    mb = 2
    U = 1.0
    V = np.zeros((n,n,n,n))
    for i in range(n):
        V[i,i,i,i] = U

    h = np.diag(np.ones(n-1),1)
    h[0,-1] = 1
    h += h.T
    h*=-1
    for i in range(0,n,2):
        h[i,i] = 1

    uhf = UHF(h,V,ma,mb)
    uhf._do_scf()

    print(uhf.F)
    eigval, eigvec = np.linalg.eigh(uhf.F)
    print(eigval)
    print(eigvec)
    Wocc = eigvec[:,:(ma+mb)]
    rho_man = Wocc.dot(Wocc.T)
    print(rho_man)



    hf_test = est.hf.HF(h,V,ma)
    hf_test._do_hf()


#    print "UHF Calc:"
#    print uhf.rho
#    print uhf.rho[:n,:n] + uhf.rho[n:,n:]
#
    print("RHF Calc:")
    print(hf_test.rho)
    print(hf_test.e)


















