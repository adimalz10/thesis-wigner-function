import numpy as np

def localized_state(N, frac=(0.35, 0.65)):

    psi = np.zeros(N, dtype=complex)

    beg = int(frac[0]*N)
    end = int(frac[1]*N)

    psi[beg:end] = (np.random.uniform(-1,1,end-beg)+ 1j*np.random.uniform(-1,1,end-beg))

    psi /= np.linalg.norm(psi)

    return psi

def q_eigenstate(N):
    
    psi = np.zeros(N, dtype = complex)

    psi[np.random.randint(0, N-1)] = 1

    return psi

def random_state(N):
    
    psi = np.random.rand(N)

    return psi/np.linalg.norm(psi)

def coherent_state(N, Z = -0.1j, p = 0.5, q = 0.5, M = 20):

    # parameters for creating the state
    kappa1 = 2*np.pi*np.random.uniform()
    kappa2 = 2*np.pi*np.random.uniform()
    hbar = 1/(2*np.pi*N)

    psi = np.zeros(N, dtype = complex)

    for j in range(N):
        cj = 0.0
        for m in range(-M, M + 1):
            xjm = j/N + kappa2/(2*np.pi*N) - m
            cj += np.exp(1j*(kappa1*m - (xjm/hbar) - (Z * (xjm-q)**2)/(2*hbar)))
            
        psi[j] = cj * np.exp(-1j*p*q/(2*hbar))
    return psi/np.linalg.norm(psi)