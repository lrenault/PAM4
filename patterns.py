import numpy as np
import matplotlib.pyplot as plt

def H_attack_decay(M, N, decay=0.01):
    """ Requirement: M < N.
    Args:
        - M (int): note length.
        - N (int): number of frames.
        - decay (float): exponential decay time
    """
    H = np.zeros((M, N))

    for n in range(N):
        # Dirac pattern
        H[M-1-n, N-1-n] = 1

        # exponential decay
        H[: N-1-n, N-1-n] = np.flip(np.exp(- decay * np.arange(N-1-n)))

    return H

def G_HMM(K, M):
    """
    Args:
        - K (int):
    """
    return None

plt.imshow(H_attack_decay(180, 80))
plt.title("H_ex for non-sustained sources")
plt.xlabel("Note length")
plt.ylabel("Number of frames")
plt.plot()
