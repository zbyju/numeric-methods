import numpy as np

from errors import *
from gauss_seidel import Gauss_Seidel
from jacobi import Jacobi


def generate_A(gamma):
  return np.array([
    [gamma, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [-1, gamma, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, -1, gamma, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, -1, gamma, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, -1, gamma, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, -1, gamma, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, -1, gamma, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, -1, gamma, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, -1, gamma, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, -1, gamma, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, gamma, -1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, gamma, -1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, gamma, -1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, gamma, -1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, gamma, -1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, gamma, -1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, gamma, -1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, gamma, -1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, gamma, -1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, gamma],
  ])

def generate_b(gamma):
  return np.array([gamma - 1, gamma - 2, gamma - 2, gamma - 2, gamma - 2, gamma - 2, gamma - 2, gamma - 2, gamma - 2, gamma - 2,
                   gamma - 2, gamma - 2, gamma - 2, gamma - 2, gamma - 2, gamma - 2, gamma - 2, gamma - 2, gamma - 2, gamma - 1])

def solve_jacobi(gamma):
  A = generate_A(gamma)
  b = generate_b(gamma)
  x0 = np.zeros(20)
  eps = 10e-6
  K = 10e8
  return Jacobi(A, b, x0, eps, K).solve()

def solve_gauss_seidel(gamma):
  A = generate_A(gamma)
  b = generate_b(gamma)
  x0 = np.zeros(20)
  eps = 10e-6
  K = 10e8
  return Gauss_Seidel(A, b, x0, eps, K).solve()

if __name__ == '__main__':
  gammas = [5, 2, 1/2]

  for gamma in gammas:
    gammaStr = str(gamma)
    try:
      print(solve_jacobi(gamma)[1])
    except DivergenceError:
      print("Jacobi " + gammaStr + " - Divergence error")
    except TooManyIterationsError:
      print("Jacobi " + gammaStr + " - Too many iterations error")

    try:
      print(solve_gauss_seidel(gamma)[1])
    except DivergenceError:
      print("GaussSeidel " + gammaStr + " - Divergence error")
    except TooManyIterationsError:
      print("GaussSeidel " + gammaStr + " - Too many iterations error")
