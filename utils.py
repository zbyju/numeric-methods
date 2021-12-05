import numpy as np


def spectral_radius(A):
  eigvals = np.linalg.eigvals(A)
  return max(map(abs, eigvals))
