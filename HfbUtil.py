import numpy as np


def assembleG(rho, Ki):
    n = rho.shape[0]
    G = np.zeros((2*n,2*n))
    G[:n,:n] = rho
    G[n:,n:] = np.eye(n)-rho
    G[:n,n:] = Ki
    G[n:,:n] = -1.*Ki
    return G


def matDirSum(mat_list):
    m_list,n_list = [0],[0]
    for i,mat in enumerate(mat_list):
        m_list.append(m_list[i]+mat.shape[0])
        n_list.append(n_list[i]+mat.shape[1])
    mat_dir_sum = np.zeros((m_list[-1],n_list[-1]))
    for i,mat in enumerate(mat_list):
        mat_dir_sum[m_list[i]:m_list[i+1],n_list[i]:n_list[i+1]] = mat
    return mat_dir_sum

        
def readFortArr(fortfile, n):
    with open(fortfile) as f:
        data = np.array(f.read().split(), dtype=float).reshape(n,n)
    return data


def loadTFCI(nf):
    Init = np.genfromtxt('Init.dat')
    n = int(Init[0])

    h = readFortArr('h.dat', n)
    A = readFortArr('A.dat', n)
    U = readFortArr('U.dat', n)
    D = readFortArr('D.dat', n)

    with open('B.dat') as f:
        B = np.array(f.read().split(), dtype=float)
    with open('V.dat') as f:
        V = np.array(f.read().split(), dtype=float).reshape(n,n,n,n)

    return n, h, V, A, U, D, B

def PKfromZ(Z):
    n = Z.shape[0]
    I = np.eye(n)
    P1 = I + np.linalg.inv(np.dot(Z,Z)-I)
    K = np.dot(Z,np.linalg.inv(np.dot(Z,Z)-I))
    return P1, K

def GfromZ(Z):
    P1, K = PKfromZ(Z)
    return genG(P1,K)

def UVfromZ(Z): #following notes from AGP_Schmidt_2
    P1, K = PKfromZ(Z)
    U,V = UVfromPK(P1, K)
    return U,V

def UVfromPK(P1, K):
    e_P1,v_P1 = np.linalg.eigh(P1)
    for i,eigval in enumerate(e_P1):
        if abs(eigval) < 10**-8:
            e_P1[i] = 0.
    v = np.diag(e_P1**0.5)
    k_no = np.dot(v_P1.T,np.dot(K,v_P1))
    u = np.dot(np.diag(np.diag(v)**-1),k_no)
    U = np.dot(v_P1,u)
    V = np.dot(v_P1,v)
#    print v
#    print
#    print k_no
#    print
#    print u
#    print
#    print U
#    print
#    print V
    return U,V

def genG(P1,K):
    n = P1.shape[0]
    I = np.eye(n)
    G = np.zeros([2*n,2*n])
    G[:n,:n] = P1
    G[:n,n:] = K
    G[n:,:n] = -K
    G[n:,n:] = I - P1
    return G
