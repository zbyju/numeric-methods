import numpy as np

from errors import *
from utils import spectral_radius


class Jacobi:
  def __init__(self, A, b, x0, eps, K):
    """
    Initialize variables needed for the Jacobi method.

    Jacobi method calculates with:
    D - Matrix A with all elements zeroed apart from the diagonal
    LU - Matrix A with the diagonal zeroed (the rest is the same)
    Dinv - The inverse matrix of matrix D

    These variables are calculated beforehand because they will not be changing
    """

    self.A = np.copy(A)
    self.b = np.copy(b)
    self.x = np.copy(x0)
    self.eps = eps
    self.K = K

    self.D = np.diag(np.diag(self.A))
    self.LU = np.subtract(self.A, self.D)
    self.Dinv = np.linalg.inv(self.D)

  def solve(self):
    """
    The main loop of Jacobi method.

    In each iteration the solution x is improved until the stopping criterium is satisfied.

    The results could be either:
      - tuple(x, i) where x is the solution and i is the number of iterations
      - DivergenceError if the method is not convergent
      - TooManyIterationsError if the number of iterations surpasses K
    """
    i = 0
    if(not self.convergence_check()):
      raise DivergenceError

    while i < self.K:
      self.x = self.next_x()
      i += 1
      if(self.stopping_criterium()):
        return (self.x, i)

    raise TooManyIterationsError


  def next_x(self):
    """
    Calculates the next x_k using the Jacobi method

    x_k = Dinv (b - (L + U) * x_(k-1) )
    """
    return np.dot(self.Dinv, (np.subtract(self.b, np.dot(self.LU, self.x))))

  def stopping_criterium(self):
    """
    Calculates the stopping criterium based on the assignment
    """
    return np.linalg.norm(np.subtract(np.dot(self.A, self.x), self.b)) / np.linalg.norm(self.b) < self.eps

  def convergence_check(self):
    """
    Checks the convergence criterium for the Jacobi method - spectral radius of (Dinv * LU) has to be < 1
    """
    return spectral_radius(np.dot(self.Dinv, self.LU)) < 1
