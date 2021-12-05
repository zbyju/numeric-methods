import numpy as np

from errors import *
from utils import *


class Gauss_Seidel:
  def __init__(self, A, b, x0, eps, K):
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
    return np.dot(self.LDinv, (np.subtract(self.b, np.dot(self.U, self.x))))

  def convergence_criterium(self):
    return np.linalg.norm(np.subtract(np.dot(self.A, self.x), self.b)) / np.linalg.norm(self.b) < self.eps

  def convergence_check(self):
    return is_pos_def(self.A) and is_symmetric(self.A)
