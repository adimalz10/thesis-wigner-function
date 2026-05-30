import numpy as np
from .floquet import floquet_kicked_harper, floquet_kicked_rotor, evolve_state
from .negativity import negativity_from_state


def negativity_time_series(N, alpha, beta, kappa, steps, state, model):

    if model.lower() == "rotor":
        U = floquet_kicked_rotor(N, alpha, beta, kappa)
    elif model.lower() == "harper":
        U = floquet_kicked_harper(N, alpha, beta, kappa)
    else:
        print("Model not found")

    assert np.allclose(U.conj().T @ U, np.eye(N), atol=1e-3), rf"Floquet operator is not unitary for N = {N}, $\alpha = {alpha}$, $\beta = {beta} and $\kappa$ = {kappa}"

    psi0 = state(N)

    neg = np.zeros(steps)

    for t in np.arange(steps):
        neg[t] = negativity_from_state(psi0)
        psi = U @ psi0
        assert abs(np.linalg.norm(psi) - 1) < 1e-3, "Wavefunction not normalized"
        psi0 = psi

    return neg


def coupling_scan(N, kappas, alpha, beta, state, model, steps=None):

    kappas = list(kappas)
    neg = np.zeros(len(kappas))

    psi0 = state(N)

    for i, kappa in enumerate(kappas):
        if model.lower() == "rotor":
            U = floquet_kicked_rotor(N, alpha, beta, kappa)
        elif model.lower() == "harper":
            U = floquet_kicked_harper(N, alpha, beta, kappa)
        else:
            print("Model not found")

        psi = evolve_state(U, psi0, steps=steps)
        neg[i] = negativity_from_state(psi)

    return neg


def system_size_scan(sizes, kappa, alpha, beta, state, model, steps=None):

    sizes = list(sizes)
    neg = np.zeros(len(sizes))

    for i, N in enumerate(sizes):
        psi0 = state(N)
        if model.lower() == "rotor":
            U = floquet_kicked_rotor(N, alpha, beta, kappa)
        elif model.lower() == "harper":
            U = floquet_kicked_harper(N, alpha, beta, kappa)
        else:
            print("Model not found")
        psi = evolve_state(U, psi0, steps=steps)
        neg[i] = negativity_from_state(psi)

    return neg


def symmetry_scan(alphas, betas, N, kappa, state, model, steps=None):

    alphas = list(alphas)
    betas = list(betas)

    neg = np.zeros((len(alphas), len(betas)))

    psi0 = state(N)

    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            if model.lower() == "rotor":
                U = floquet_kicked_rotor(N, alpha, beta, kappa)
            elif model.lower() == "harper":
                U = floquet_kicked_harper(N, alpha, beta, kappa)
            else:
                print("Model not found")
            psi = evolve_state(U, psi0, steps=steps)
            neg[i, j] = negativity_from_state(psi)

    return neg