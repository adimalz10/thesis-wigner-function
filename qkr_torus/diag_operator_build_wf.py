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

    # create the floquet operator

    # first step: make the DFT vector and perform a DFT

    m = np.arange(N)
    dm = np.exp((-1j * np.pi * (m + beta)**2)/N)

    f = scp.fft.ifft(dm)

    # next step is to build the operator now

    n = m.copy()
    kn = np.exp((1j * N * K)/(2*np.pi) * np.cos(2*np.pi*(n+alpha)/N)) # kick operator
    for i in range(N):
        for k in range(N):
            U[i, k] = kn[i] * np.exp((1j * 2 * np.pi/N)*beta*((i-k)%N)) * f[(i-k)%N]

    return U

def wigner_function(psi):
    """
    psi: wavefunction
    """

    psi = psi.reshape(-1)
    N = len(psi)
    rho = np.outer(psi, psi.conj())      # density matrix

    # indices k, l
    k = np.arange(N).reshape(N,1)
    l = np.arange(N).reshape(1,N)

    # all possible (k+l) mod N
    S = (k + l) % N

    # all possible (k-l) mod N
    D = (k - l) % N

    # precompute rho(k,l) once
    R = rho

    # output Wigner function
    W = np.zeros((N, N), dtype=complex)

    # For each a1, pick out the stripe where k+l = 2*a1
    for a1 in range(N):
        mask = (S == (2*a1) % N)              # boolean mask for that diagonal stripe
        vals = np.zeros(N, dtype=complex)     # values for all a2 for this a1

        # pick the (k,l) pairs in this stripe
        R_stripe = R[mask]                    # these are rho(k,l)
        D_stripe = D[mask]                    # these are (k-l)

        # compute FFT along a2 direction:
        # sum over stripe entries: rho(k,l) * exp(2πi a2 (k-l)/N)
        # which is exactly the DFT of rho(k,l) grouped by (k-l)
        # so build an accumulator of size N for each (k-l)
        tmp = np.zeros(N, dtype=complex)
        for r, d in zip(R_stripe, D_stripe):
            tmp[d] += r

        # Now vals[a2] = tmp_d * exp(2π i a2 d / N)
        W[a1] = (1/N) * np.fft.fft(tmp)

    return W.real