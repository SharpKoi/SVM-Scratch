import numpy as np


def linear_kernel(x: np.array,
                  y: np.array):
    return np.dot(x, y)


def poly_kernel(x: np.array,
                y: np.array):
    return np.square(np.dot(x, y)+1)


def gaussian_kernel(x: np.array,
                    y: np.array):
    return np.exp(-np.square(np.linalg.norm(x-y)) / 2)
