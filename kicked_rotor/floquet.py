import numpy as np

def floquet_kicked_rotor(N, alpha, beta, K):

    n = np.arange(N)
    m = np.arange(N)

    kick = np.exp(1j * N * K/(2*np.pi) * np.cos(2 * np.pi * (n + alpha) / N))

    d = np.exp(-1j * np.pi * (m + beta)**2 / N)
    f = np.fft.ifft(d)  

    diff = n[:, None] - n[None, :]        
    idx  = diff % N                        

    free = f[idx] * np.exp(2j * np.pi * beta * diff / N)   

    U = kick[:, None] * free

    return U

def floquet_kicked_harper(N, alpha, beta, K):
    n = np.arange(N)
    m = np.arange(N)

    kick = np.exp(1j * N * K * np.cos(2 * np.pi * (n + alpha) / N))

    d = np.exp(-1j * N * np.cos((2*np.pi/N)*(m+beta)))
    f = np.fft.ifft(d)  

    diff = n[:, None] - n[None, :]        
    idx  = diff % N                        

    free = f[idx] * np.exp(2j * np.pi * beta * diff / N)   

    U = kick[:, None] * free

    return U


def evolve_state(U, psi, steps = None):

    if steps is None:
        steps = U.shape[0]  # heisenberg time

    if steps == 0:
        return psi

    for _ in range(steps):
        psi = U @ psi
        assert abs(np.linalg.norm(psi) - 1)<1e-3, "Wavefunction not normalized"

    return psi