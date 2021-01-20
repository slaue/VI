# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import multivariate_normal

def generate_data(n=100):
    np.random.seed(0)
    shifted_gaussian = np.random.randn(n, 2) + np.array([3, 3])

    C = np.array([[0., -0.7],
                  [3.5, .7]])
    stretched_gaussian = np.dot(np.random.randn(n, 2), C)

    X = np.vstack([shifted_gaussian,
                   stretched_gaussian])
    y = np.hstack((0*np.ones(n, dtype=np.int), np.ones(n, dtype=np.int)))
    return X, y

def plot(data, mu=None, sigma=None, title='', y=None):
    colors = ['red', 'blue']

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')

    if y is None:
        ax.scatter(data[:, 0], data[:, 1], s=20, color='black')
    else:
        for n, color in enumerate(colors):
            plot_data = data[y == n]
            plt.scatter(plot_data[:, 0], plot_data[:, 1], s=20, color=color)

    if not mu is None:
        x1_min, x1_max, x2_min, x2_max = ax.axis()
        X1, X2 = np.mgrid[x1_min:x1_max:.1, x2_min:x2_max:.1]
        pos = np.dstack((X1, X2))

        rv_0 = multivariate_normal(mu[0], sigma[0])
        vals_0 = rv_0.pdf(pos)
        rv_1 = multivariate_normal(mu[1], sigma[1])
        vals_1 = rv_1.pdf(pos)
        vals = vals_0 - vals_1
        y_min = np.min(vals)
        y_max = np.max(vals)
        y_levels = np.hstack([np.linspace(y_min, 0, 4, endpoint=False), np.linspace(0, y_max, 5)])
        ax.contourf(X1, X2, vals, alpha=0.5, levels=y_levels, cmap=cm.bwr)
        ax.contour(X1, X2, vals, levels=[0], colors='black', linewidths=4)

    plt.title(title)
    plt.show()


def compute_Gaussians(X, y):

    # Please make this correct!

    mu_0 = np.array([5, 0])
    mu_1 = np.array([-5, 2])
    mu = np.array([mu_0, mu_1])

    sigma_0 = np.eye(2)
    sigma_1 = np.eye(2)
    sigma = np.array([sigma_0, sigma_1])
    return mu, sigma


np.random.seed(0)
X, y = generate_data(100)
plot(X, y=y)
mu, sigma = compute_Gaussians(X, y)
plot(X, mu=mu, sigma=sigma, y=y)
