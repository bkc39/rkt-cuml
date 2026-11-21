import numpy as np


def host(a):
    return np.asarray(a.get() if hasattr(a, "get") else a)
