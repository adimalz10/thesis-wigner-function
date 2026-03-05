import numpy as np


def wigner_function(psi):

    rho = psi[:,None] * psi.conj()[None,:]

    N = len(rho[0])

    k = np.arange(N).reshape(N,1)
    l = np.arange(N).reshape(1,N)

    S = (k + l) % N
    D = (k - l) % N

    W = np.zeros((N,N), dtype=complex)

    for a1 in range(N):

        mask = (S == (2*a1) % N)

        R_stripe = rho[mask]
        D_stripe = D[mask]

        tmp = np.zeros(N, dtype=complex)

        for r,d in zip(R_stripe, D_stripe):
            tmp[d] += r

        W[a1] = np.fft.ifft(tmp)

    return W.real


def density_matrix(W):

    N = W.shape[0]
    rho = np.zeros((N,N), dtype=complex)

    for b in range(N):

        tmp = np.fft.fft(W[b])

        for d in range(N):

            k = (b + d * pow(2,-1,N)) % N
            l = (2*b - k) % N

            rho[k,l] = tmp[d]

    return rho