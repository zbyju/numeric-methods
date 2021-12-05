import numpy as np


def spectral_radius(A):
  """Get spectral radius for a given matrix A"""
  eigvals = np.linalg.eigvals(A)
  return max(map(abs, eigvals))

def is_symmetric(A, tol=1e-8):
  """Find out if matrix A is symmetric"""
  return np.all(np.abs(A-A.T) < tol)

def is_pos_def(A):
  """Find out if matrix A is positive definite"""
  return np.all(np.linalg.eigvals(A) > 0)
