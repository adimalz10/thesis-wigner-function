import numpy as np
import scipy as scp


"""def diagonalize_floquet_operator(N, alpha, beta, K):

    U = np.zeros((N,N), dtype=complex)

    m = np.arange(N)
    dm = np.exp((-1j*np.pi*(m+beta)**2)/N)

    f = scp.fft.ifft(dm, norm = 'ortho') / np.sqrt(N)

    n = m.copy()
    kn = np.exp((1j*N*K)/(2*np.pi) * np.cos(2*np.pi*(n+alpha)/N))

    for i in range(N):
        for k in range(N):
            U[i,k] = (
                kn[i]
                * np.exp((1j*2*np.pi/N)*beta*((i-k)%N))
                * f[(i-k)%N]
            )

    return U"""

def diagonalize_floquet_operator(N, alpha, beta, K):

    n = np.arange(N)
    m = np.arange(N)

    kick = np.exp(-2j * np.pi * K / N * np.cos(2 * np.pi * (n + alpha) / N))

    d = np.exp(-1j * np.pi * (m + beta)**2 / N)
    f = np.fft.ifft(d)  # f[k] = (1/N) * sum_m d_m * exp(2pi*i*m*k/N)

    diff = n[:, None] - n[None, :]        # actual (n - n'), shape (N, N), not reduced mod N
    idx  = diff % N                        # used only for indexing into f

    free = f[idx] * np.exp(2j * np.pi * beta * diff / N)   # diff, not idx

    U = kick[:, None] * free

    return U


def evolve_state(U, psi, steps = None):

    if steps is None:
        steps = U.shape[0]  # heisenberg time

    if steps == 0:
        return psi

    for _ in range(steps):
        psi = U @ psi

    return psi