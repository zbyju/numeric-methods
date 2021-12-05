import numpy as np


def spectral_radius(A):
  eigvals = np.linalg.eigvals(A)
  return max(map(abs, eigvals))

def is_symmetric(A, tol=1e-8):
  return np.all(np.abs(A-A.T) < tol)

def is_pos_def(A):
  return np.all(np.linalg.eigvals(A) > 0)
