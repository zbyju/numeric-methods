import numpy as np

from jacobi import Jacobi

if __name__ == '__main__':
    A = np.array([[1, 0], [0, 1]])
    b = np.array([3, 4])
    x0 = np.array([10, 10])
    eps = 10e-9

    js = Jacobi(A, b, x0, eps, 10000000)

    print(js.solve())
