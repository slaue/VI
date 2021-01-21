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

    C = np.array([[0., 0.7],
                  [-2, .7]])
    stretched_gaussian2 = np.dot(np.random.randn(n, 2), C) + np.array([-5, 3])

    data = np.vstack([shifted_gaussian,
                      stretched_gaussian,
                      stretched_gaussian2]) / 5 + np.array([0, -0.5])
    return data

def plot(data, mu=None, sigma=None, title=''):
    cmap = [cm.Blues, cm.Greens, cm.Reds, cm.Oranges]

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')

    ax.scatter(data[:, 0], data[:, 1], s=20, color='black')

    if not mu is None:
        x1_min, x1_max, x2_min, x2_max = ax.axis()
        X1, X2 = np.mgrid[x1_min:x1_max:.1, x2_min:x2_max:.1]
        pos = np.dstack((X1, X2))

        for i in [0, 1, 2]:
            rv = multivariate_normal(mu[i][:2], sigma[i][:2, :2])
            vals = rv.pdf(pos)
            y_max = np.max(vals)
            ax.contourf(X1, X2, vals, levels=[0.2*y_max, 0.4*y_max, 0.6*y_max, 0.8*y_max, y_max], alpha=0.5, cmap=cmap[i]) # , linewidths=3)

    plt.title(title)
    plt.show()


def E_step(p, mu, sigma, data):

    ### Please fill in here!

    return gamma

def M_step(gamma, data):

    ### Please fill in here!

    return p, mu, sigma

def EM(data, n_classes, iters=0, show=False):
    # start with random p, random means, unit variance
    d = data.shape[1]
    p = np.random.rand(n_classes)
    p /= p.sum()
    mu = np.random.randn(n_classes, d)
    sigma = np.array([0.1*np.eye(d) for i in range(n_classes)])

    if show:
        plot(data, mu, sigma, title='iteration 0')
    for i in range(iters):
        gamma = E_step(p, mu, sigma, data)
        p, mu, sigma = M_step(gamma, data)
        if show:
            plot(data, mu, sigma, title=f'iteration {i+1}')

    return p, mu, sigma


np.random.seed(0)
data = generate_data(100)
plot(data)

p, mu, sigma = EM(data, n_classes=3, iters=0, show=True)
