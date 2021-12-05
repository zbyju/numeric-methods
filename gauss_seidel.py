import numpy as np

from errors import *
from utils import *


class Gauss_Seidel:
  def __init__(self, A, b, x0, eps, K):
    """
    Initialize variables needed for the Gauss-Seidel method.

    Gauss-Seidel method calculates with:
    D - Matrix A with all elements zeroed apart from the diagonal
    L - Matrix A with all elemnts on the diagonal and above it zeroed
    U - Matrix A with all elemnts on the diagonal and below it zeroed

    (It holds that A = L + D + U)

    LD - Matrix L + D
    LDinv - The inverse matrix of matrix LD

    These variables are calculated beforehand because they will not be changing
    """
    self.A = np.copy(A)
    self.b = np.copy(b)
    self.x = np.copy(x0)
    self.eps = eps
    self.K = K

    self.D = np.diag(np.diag(self.A))
    self.L = np.tril(self.A, -1)
    self.U = np.triu(self.A, 1)
    self.LD = np.copy(np.add(self.L, self.D))
    self.LDinv = np.linalg.inv(self.LD)

  def solve(self):
    """
    The main loop of Gauss-Seidel method.

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
      if(self.convergence_criterium()):
        return (self.x, i)

    raise TooManyIterationsError


  def next_x(self):
    """
    Calculates the next x_k using the Gauss-Seidel method

    x_k = LDinv * (b - (U * x_(k-1)) )
    """
    return np.dot(self.LDinv, (np.subtract(self.b, np.dot(self.U, self.x))))

  def convergence_criterium(self):
    """
    Calculates the stopping criterium based on the assignment
    """
    return np.linalg.norm(np.subtract(np.dot(self.A, self.x), self.b)) / np.linalg.norm(self.b) < self.eps

  def convergence_check(self):
    """
    Checks the convergence criterium for the Gauss-Seidel method:
      - Matrix A is symmetric
      - Matrix A is positive definite
    """
    return is_symmetric(self.A) and is_pos_def(self.A)
