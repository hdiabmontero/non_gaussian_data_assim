import numpy as np


def randsample(n: int, p: np.ndarray) -> np.ndarray:
    """
    Perform resampling based on given probabilities.

    Args:
    n (int): Number of items to sample.
    p (numpy.array): Array of probabilities associated with each item.

    Returns:
    numpy.array: Array of indices, sampled according to probabilities p.
    """
    return np.random.choice(np.arange(0, n, 1), size=n, replace=True, p=p)
