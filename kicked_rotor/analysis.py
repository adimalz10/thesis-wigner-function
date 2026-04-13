import numpy as np
from tqdm import tqdm

from .states import localized_state
from .floquet import diagonalize_floquet_operator, evolve_state
from .negativity import negativity_from_state


def negativity_time_series(N, alpha, beta, kappa, steps, state):
    """
    Negativity vs time for fixed parameters.
    """
    U = diagonalize_floquet_operator(N, alpha, beta, kappa)

    assert np.allclose(U.conj().T @ U, np.eye(N), atol=1e-3) , rf"Floquet operator is not unitary for N = {N}, $\alpha = {alpha}$, $\beta = {beta} and $\kappa$ = {kappa}"

    psi0 = state(N)

    neg = np.zeros(steps)

    for t in tqdm(np.arange(steps), desc="coupling", leave = False):
        neg[t] = negativity_from_state(psi0)
        psi = U @ psi0
        assert abs(np.linalg.norm(psi) - 1)<1e-3, "Wavefunction not normalized"
        psi0 = psi

    return neg


def coupling_scan(N, kappas, alpha, beta, state, steps = None):
    """
    Negativity vs coupling strength.
    """
    kappas = list(kappas)
    neg = np.zeros(len(kappas))

    psi0 = state(N)

    for i, kappa in enumerate(tqdm(kappas, desc="coupling", leave = False)):

        U = diagonalize_floquet_operator(N, alpha, beta, kappa)
        psi = evolve_state(U, psi0, steps = steps)
        neg[i] = negativity_from_state(psi)

    return neg


def system_size_scan(sizes, kappa, alpha, beta, state, steps = None):
    
    sizes = list(sizes)
    neg = np.zeros(len(sizes))

    for i, N in enumerate(tqdm(sizes, desc="system size", leave = False)):
        psi0 = state(N)
        U = diagonalize_floquet_operator(N, alpha, beta, kappa)
        psi = evolve_state(U, psi0, steps = steps)
        neg[i] = negativity_from_state(psi)

    return neg

"""def system_size_scan(sizes, kappa, alpha, beta, samples = 80):

    sizes = list(sizes)
    neg = np.zeros(len(sizes))

    for i, N in enumerate(tqdm(sizes, desc="system size")):

        U = diagonalize_floquet_operator(N, alpha, beta, kappa)
        psi = localized_state(N)

        tH = N
        dt = max(1, tH // samples)

        neg_sum = 0
        count = 0

        for t in range(tH):

            if t % dt == 0:
                neg_sum += negativity_from_state(psi)
                count += 1

            psi = U @ psi

        neg[i] = neg_sum / count

    return neg"""


def symmetry_scan(alphas, betas, N, kappa, state, steps = None):
    """
    Negativity after time evolution for a grid of (alpha, beta).

    Returns
    -------
    neg : ndarray (len(alphas), len(betas))
        Negativity for each symmetry parameter pair.
    """

    alphas = list(alphas)
    betas = list(betas)

    neg = np.zeros((len(alphas), len(betas)))

    pbar = tqdm(total=len(alphas) * len(betas), desc="(alpha,beta) scan", leave = False)

    psi0 = state(N)

    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):

            U = diagonalize_floquet_operator(N, alpha, beta, kappa)

            psi = evolve_state(U, psi0, steps = steps)

            neg[i, j] = negativity_from_state(psi)

            pbar.update(1)

    pbar.close()

    return neg