import unittest

import numpy as np

from gauss_seidel import Gauss_Seidel
from jacobi import Jacobi
from utils import spectral_radius


class TestUtils(unittest.TestCase):
  def test_spectral_radius(self):
    E = np.array([[1, 0], [0, 1]])
    A1 = np.array([[3, 0], [0, 3]])
    A2 = np.array([[-3, 0], [0, -3]])
    A3 = np.array([[-3, 0], [0, 0]])

    self.assertEqual(1, spectral_radius(E))
    self.assertEqual(3, spectral_radius(A1))
    self.assertEqual(3, spectral_radius(A2))
    self.assertEqual(3, spectral_radius(A3))

  def test_jacobi(self):
    E = np.array([[1, 0], [0, 1]])
    A1 = np.array([[1, 2], [1, 3]])
    A2 = np.array([[-101, 100], [-101, 101]])
    b = np.array([3, 4])
    x0 = np.array([10, 10])

    je = Jacobi(E, b, x0, 10e-9, 10e7).solve()
    ja1 = Jacobi(A1, b, x0, 10e-9, 10e7).solve()
    ja2 = Jacobi(A2, b, x0, 10e-9, 10e7).solve()

    self.assertTrue(np.allclose(np.array([3, 4]), je[0]))
    self.assertTrue(np.allclose(np.array([1, 1]), ja1[0]))
    self.assertTrue(np.allclose(np.array([97/101, 1]), ja2[0]))

  def test_gauss_seidel(self):
    E = np.array([[1, 0], [0, 1]])
    b = np.array([3, 4])
    x0 = np.array([10, 10])

    ge = Gauss_Seidel(E, b, x0, 10e-9, 10e7).solve()

    self.assertTrue(np.allclose(np.array([3, 4]), ge[0]))

if __name__ == '__main__':
    unittest.main()
