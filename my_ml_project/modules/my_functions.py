import numpy as np

def sum_of_squares(n: int) -> int:
    """
    Compute the sum of squares of numbers from 1 to n.
    """
    if n < 1:
        raise ValueError("n must be at least 1.")
    return np.sum(np.arange(1, n+1) ** 2)
