import numpy as np
from cvxopt import matrix
from cvxopt import solvers
import matplotlib.pyplot as plt

from kernel import linear_kernel, poly_kernel, gaussian_kernel


class SVM:
    def __init__(self, kernel='linear'):
        self.kernel = {'linear': linear_kernel,
                       'poly': poly_kernel,
                       'gaussian': gaussian_kernel}[kernel]
        self.weight = 0
        self.bias = 0
        self.opt_coef = np.array([])
        self.initialized = False

    def fit(self,
            features: np.array,
            labels: np.array):
        self.x = x = features
        self.y = y = labels

        n_samples, n_features = x.shape
        # generate necessary arguments for cvxopt qp solver
        P = matrix([[y[i]*y[j]*self.kernel(x[i], x[j]) for j in range(n_samples)] for i in range(n_samples)])
        q = matrix(-np.ones((n_samples, 1)))
        G = matrix(-np.eye(n_samples))
        h = matrix(np.zeros(n_samples))
        A = matrix(y.reshape(1, -1))
        b = matrix(np.zeros(1))

        # config solver
        # optimize the optimal coefficients
        sol = solvers.qp(P, q, G, h, A, b)
        self.opt_coef = opt_coef = np.array(sol['x']).flatten()

        # calculate weight and bias
        self.weight = (opt_coef*y) @ x
        self.bias = (y[opt_coef > 1e-4] - np.dot(x[opt_coef > 1e-4], self.weight)).min()

        # update status
        self.initialized = True

    def hypothesis(self, x):
        if self.initialized:
            return (self.opt_coef * self.y) @ np.array(
                        [self.kernel(xx, x) for xx in self.x]) + self.bias
        else:
            print('Please train this model with some data first.')
            return None

    def plot(self):
        resolution = 50
        dx = np.linspace(-1, 1, resolution)
        dy = np.linspace(-1, 1, resolution)
        dx, dy = np.meshgrid(dx, dy)
        grid_x = np.c_[dx.flatten(), dy.flatten()]
        z = np.array([self.hypothesis(x) for x in grid_x])
        z = z.reshape(dx.shape)

        # fill the area between level curves h(x)=-10000,-1,0,1,10000 with the given color
        plt.figure(figsize=(16, 12))
        plt.contourf(dx, dy, z, [-10000, -1, 0, 1, 10000], colors=['skyblue', '#009dc4', 'sandybrown', 'bisque'])
        plt.scatter(self.x[self.y >= 0, 0], self.x[self.y >= 0, 1], s=150, color='r')
        plt.scatter(self.x[self.y < 0, 0], self.x[self.y < 0, 1], s=150, color='b')
        plt.show()


class SVC:
    def __init__(self, C=1., kernel='linear'):
        self.C = C
        self.kernel = {'linear': linear_kernel,
                       'poly': poly_kernel,
                       'gaussian': gaussian_kernel}[kernel]
        self.weight = 0
        self.bias = 0
        self.opt_coef = np.array([])
        self.initialized = False

    def fit(self,
            features: np.array,
            labels: np.array):
        self.x = x = features
        self.y = y = labels

        n, m = x.shape

        # define necessary arguments for cvxopt qp solver
        P = matrix([[y[i] * y[j] * self.kernel(x[i], x[j]) for j in range(n)] for i in range(n)])
        q = matrix(-np.ones((n, 1)))
        G = matrix(np.vstack((-np.eye(n), np.eye(n))))
        h = matrix(np.hstack((np.zeros(n), np.ones(n) * self.C)))
        A = matrix(y.reshape((1, -1)))
        b = matrix(np.zeros(1))

        # config solver
        # optimize the optimal coefficients
        sol = solvers.qp(P, q, G, h, A, b)
        self.opt_coef = opt_coef = np.array(sol['x']).flatten()

        # calc weight, bias and support vectors
        self.sv_indices = (opt_coef > 1e-4)
        self.support_vectors = x[self.sv_indices]
        self.weight = self._calc_weight()
        self.bias = self._calc_bias()

        self.initialized = True

    def _calc_weight(self):
        return (self.opt_coef[self.sv_indices] * self.y[self.sv_indices]) @ self.support_vectors

    def _calc_bias(self):
        cond = (self.y == 1) & self.sv_indices
        bias = np.Inf
        for i in np.where(cond)[0]:
            wx = self.opt_coef[self.sv_indices] * self.y[self.sv_indices] @ np.array(
                [self.kernel(sv, self.x[i]) for sv in self.support_vectors])
            bias = min(bias, self.y[i] - wx)

        return bias

    def hypothesis(self, x):
        if self.initialized:
            return (self.opt_coef[self.sv_indices] * self.y[self.sv_indices]) @ np.array(
                [self.kernel(sv, x) for sv in self.support_vectors]) + self.bias
        else:
            print('Please train this model with some data first.')
            return None

    def plot(self):
        resolution = 50
        dx = np.linspace(-1, 1, resolution)
        dy = np.linspace(-1, 1, resolution)
        dx, dy = np.meshgrid(dx, dy)
        grid_x = np.c_[dx.flatten(), dy.flatten()]
        z = np.array([self.hypothesis(x) for x in grid_x])
        z = z.reshape(dx.shape)
        plt.figure(figsize=(16, 12))
        # fill the area between level curves h(x)=-10000,-1,0,1,10000 with the given color
        plt.contourf(dx, dy, z, [-10000, -1, 0, 1, 10000], colors=['skyblue', '#009dc4', 'sandybrown', 'bisque'])
        plt.scatter(self.x[self.y >= 0, 0], self.x[self.y >= 0, 1], s=150, color='r')
        plt.scatter(self.x[self.y < 0, 0], self.x[self.y < 0, 1], s=150, color='b')
        plt.show()