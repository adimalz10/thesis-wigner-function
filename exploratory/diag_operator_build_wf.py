import numpy as np
import scipy as scp

def diagonalize_floquet_operator(N, alpha, beta, K):
    """
    N: number of dimensions in the Hilbert Space
    alpha: breaking of parity
    beta: time reversal symmetry
    K: kicking strength
    """

    U = np.zeros((N, N), dtype = complex)

    m = np.arange(N)
    dm = np.exp((-1j * np.pi * (m + beta)**2)/N)

    f = scp.fft.ifft(dm)

    n = m.copy()
    kn = np.exp((1j * N * K)/(2*np.pi) * np.cos(2*np.pi*(n+alpha)/N)) # kick operator
    for i in range(N):
        for k in range(N):
            U[i, k] = kn[i] * np.exp((1j * 2 * np.pi/N)*beta*((i-k)%N)) * f[(i-k)%N]

    return U

def wigner_function(rho):
    """
    rho: density matrix
    """

    N = len(rho[0])

    # indices k, l
    k = np.arange(N).reshape(N,1)
    l = np.arange(N).reshape(1,N)
    inv2 = pow(2, -1, N)

    # all possible (k+l) mod N
    S = (k + l) % N

    # all possible (k-l) mod N
    D = ((k - l)) % N

    # precompute rho(k,l) once
    R = rho

    # output Wigner function
    W = np.zeros((N, N), dtype=complex)

    for a1 in range(N):
        mask = (S == (2*a1) % N)              
        vals = np.zeros(N, dtype=complex)     

        
        R_stripe = R[mask]                    
        D_stripe = D[mask]                    

        tmp = np.zeros(N, dtype=complex)
        for r, d in zip(R_stripe, D_stripe):
            tmp[d] += r

        # Now vals[a2] = tmp_d * exp(2π i a2 d / N)
        W[a1] = np.fft.ifft(tmp)

    return W
    

def density_matrix(W):
    
    N = W.shape[0]
    rho = np.zeros((N, N), dtype=complex)

    for b in range(N):
    
        tmp = np.fft.fft(W[b])

        for d in range(N):
            # solve k - l = d, k + l = 2b
            k = (b + d * pow(2, -1, N)) % N
            l = (2*b - k) % N
            rho[k, l] = tmp[d]

    return rho