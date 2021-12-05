import numpy as np

from errors import *
from utils import spectral_radius


class Jacobi:
  def __init__(self, A, b, x0, eps, K):
    self.A = np.copy(A)
    self.b = np.copy(b)
    self.x = np.copy(x0)
    self.eps = eps
    self.K = K

    self.D = np.diag(np.diag(self.A))
    self.LU = np.subtract(self.A, self.D)
    self.Dinv = np.linalg.inv(self.D)

  def solve(self):
    i = 0
    if(not self.convergence_check()):
      raise DivergenceError

    while i < self.K:
      self.x = self.next_x()
      i += 1
      if(np.linalg.norm(np.subtract(np.matmul(self.A, self.x), self.b)) < self.eps):
        return (self.x, i)

    raise TooManyIterationsError


  def next_x(self):
    return np.dot(self.Dinv, (np.subtract(self.b, np.dot(self.LU, self.x))))

  def convergence_criterium(self):
    c = np.linalg.norm(np.subtract(np.dot(self.A, self.x), self.b)) / np.linalg.norm(self.b) < self.eps
    if(np.isinf(c)):
      raise DivergenceError

  def convergence_check(self):
    return spectral_radius(np.dot(self.Dinv, self.LU)) < 1
