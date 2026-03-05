import numpy as np
from tqdm import tqdm

from .states import localized_state
from .floquet import diagonalize_floquet_operator, evolve_state
from .negativity import negativity_from_state


def negativity_time_series(N, alpha, beta, kappa, steps):
    """
    Negativity vs time for fixed parameters.
    """
    U = diagonalize_floquet_operator(N, alpha, beta, kappa)

    psi = localized_state(N)

    neg = np.zeros(steps)

    for t in range(steps):
        neg[t] = negativity_from_state(psi)
        psi = U @ psi

    return neg


def coupling_scan(N, kappas, alpha, beta, steps=10):
    """
    Negativity vs coupling strength.
    """
    kappas = list(kappas)
    neg = np.zeros(len(kappas))

    for i, kappa in enumerate(tqdm(kappas, desc="coupling")):

        U = diagonalize_floquet_operator(N, alpha, beta, kappa)

        psi = localized_state(N)
        psi = evolve_state(U, psi, steps)

        neg[i] = negativity_from_state(psi)

    return neg


def system_size_scan(sizes, kappa, alpha, beta, steps=10):
    """
    Negativity vs Hilbert space dimension.
    """
    sizes = list(sizes)
    neg = np.zeros(len(sizes))

    for i, N in enumerate(tqdm(sizes, desc="system size")):

        U = diagonalize_floquet_operator(N, alpha, beta, kappa)

        psi = localized_state(N)
        psi = evolve_state(U, psi, steps)

        neg[i] = negativity_from_state(psi)

    return neg


def symmetry_scan(alphas, betas, N, kappa, steps):
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

    pbar = tqdm(total=len(alphas) * len(betas), desc="(alpha,beta) scan")

    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):

            U = diagonalize_floquet_operator(N, alpha, beta, kappa)

            psi = localized_state(N)
            psi = evolve_state(U, psi, steps)

            neg[i, j] = negativity_from_state(psi)

            pbar.update(1)

    pbar.close()

    return neg