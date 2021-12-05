import numpy as np

from errors import *
from gauss_seidel import Gauss_Seidel
from jacobi import Jacobi


def generate_A(gamma):
  """
  Generate the matrix as described in the assignment based on inputed gamma
  """
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
  """
  Generate the right side as described in the assignment based on inputed gamma
  """
  return np.array([gamma - 1, gamma - 2, gamma - 2, gamma - 2, gamma - 2, gamma - 2, gamma - 2, gamma - 2, gamma - 2, gamma - 2,
                   gamma - 2, gamma - 2, gamma - 2, gamma - 2, gamma - 2, gamma - 2, gamma - 2, gamma - 2, gamma - 2, gamma - 1])

def solve_jacobi(gamma):
  """
  Get solution (tuple (x, # of iterations)) using jacobi method
  """
  A = generate_A(gamma)
  b = generate_b(gamma)
  x0 = np.zeros(20)
  eps = 10e-6
  K = 10e8
  return Jacobi(A, b, x0, eps, K).solve()

def solve_gauss_seidel(gamma):
  """
  Get solution (tuple (x, # of iterations)) using gauss-seidel method
  """
  A = generate_A(gamma)
  b = generate_b(gamma)
  x0 = np.zeros(20)
  eps = 10e-6
  K = 10e8
  return Gauss_Seidel(A, b, x0, eps, K).solve()

if __name__ == '__main__':
  """
  Print all the solutions for the assignment
  """
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
