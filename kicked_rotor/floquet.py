import numpy as np
import scipy as scp


def diagonalize_floquet_operator(N, alpha, beta, K):

    U = np.zeros((N,N), dtype=complex)

    m = np.arange(N)
    dm = np.exp((-1j*np.pi*(m+beta)**2)/N)

    f = scp.fft.ifft(dm)

    n = m.copy()
    kn = np.exp((1j*N*K)/(2*np.pi) * np.cos(2*np.pi*(n+alpha)/N))

    for i in range(N):
        for k in range(N):
            U[i,k] = (
                kn[i]
                * np.exp((1j*2*np.pi/N)*beta*((i-k)%N))
                * f[(i-k)%N]
            )

    return U


def evolve_state(U, psi0, steps):

    psi = psi0
    if steps != 0:
        for _ in range(steps):
            psi = U @ psi0
            psi0 = psi
        return psi

    return psi0