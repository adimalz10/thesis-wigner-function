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

    psi[np.random.randint(N)] = 1

    return psi

def random_state(N):
    
    psi = np.random.normal(0, 1, N) + 1j*np.random.normal(0, 1, N)

    return psi/np.linalg.norm(psi)

def coherent_state(N, alpha = 0, p = 0, q = 0):
    H = np.zeros((N, N))
    n = np.arange(N)
    diag = 2 - np.cos(2 * np.pi * (n + alpha) / N)
    H = np.diag(diag) - 0.5 * (np.diag(np.ones(N-1), 1) + np.diag(np.ones(N-1), -1))
    H[0, N-1] = -0.5
    H[N-1, 0] = -0.5
    
    _ , eigvecs = np.linalg.eigh(H)
    ground_state = eigvecs[:, 0]
    shifted_state = np.zeros(N, dtype = complex)
    for j in range(N):
        shifted_state[j] = np.exp(2j * np.pi * p * (j + alpha)) * ground_state[(j - q)%N]

    return shifted_state/np.linalg.norm(shifted_state)

"""def coherent_state(N, Z = -0.1j, p = 0.5, q = 0.5, M = 20):

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
    return psi/np.linalg.norm(psi)"""