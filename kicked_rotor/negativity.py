import numpy as np
from .wigner import wigner_function


def negativity_from_state(psi):

    W = wigner_function(psi).real

    return 0.5*np.sum(np.abs(W) - W)  