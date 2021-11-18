import numpy as np


def linear_kernel(x: np.array,
                  y: np.array):
    return np.dot(x, y)


def poly_kernel(x: np.array, y: np.array, degree: int, coef0: float):
    return np.power(np.dot(x, y)+coef0, degree)


def gaussian_kernel(x: np.array, y: np.array, variance=1.):
    return np.exp(-np.square(np.linalg.norm(x-y)) / 2*variance)
