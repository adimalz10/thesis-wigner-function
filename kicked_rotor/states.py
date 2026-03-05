import numpy as np

def localized_state(N, frac=(0.35, 0.65)):

    psi = np.zeros(N, dtype=complex)

    beg = int(frac[0]*N)
    end = int(frac[1]*N)

    psi[beg:end] = (np.random.uniform(-1,1,end-beg)+ 1j*np.random.uniform(-1,1,end-beg))

    psi /= np.linalg.norm(psi)

    return psi