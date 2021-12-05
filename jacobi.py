import numpy as np

from errors import DivergenceError
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
    while i < self.K:
      self.x = self.next_x()
      if((np.linalg.norm(np.matmul(self.A, self.x) - self.b)) / np.linalg.norm(self.b) < self.eps):
        return (self.x, i)

      if(not self.convergence_check()):
        raise DivergenceError

      i += 1


  def next_x(self):
    return np.dot(self.Dinv, (np.subtract(self.b, np.dot(self.LU, self.x))))

  def convergence_check(self):
    return spectral_radius(np.dot(self.Dinv, self.LU)) < 1
