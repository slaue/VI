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


    fig = plt.figure(figsize=(8, 8))
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
            ax.contourf(X1, X2, rv.pdf(pos), levels=[0.2*y_max, 0.4*y_max, 0.6*y_max, 0.8*y_max, y_max], alpha=0.5, cmap=cmap[i]) # , linewidths=3)

    plt.title(title)
    plt.show()


def E_step(mu, sigma, X_train):
    n_classes = mu.shape[0]
    rv = [multivariate_normal(mu[i], sigma[i]) for i in range(n_classes)]
    p = np.array([rv[i].pdf(X_train) for i in range(n_classes)])

    ### Please fill in here!

    return p

def M_step(p, X_train):
    n_classes = p.shape[0]

    ### Please fill in here!

    return mu, sigma

def EM(data, n_classes, iters=0, show=False):
    # start with random means
    d = data.shape[1]
    p = np.zeros(n_classes)
    mu = np.random.randn(n_classes, d)
    sigma = np.array([0.1*np.eye(d) for i in range(n_classes)])

    if show:
        plot(data, mu, sigma, title='iteration 0')
    for i in range(iters):
        p = E_step(mu, sigma, data)
        mu, sigma = M_step(p, data)
        if show:
            plot(data, mu, sigma, title=f'iteration {i+1}')

    return p, mu, sigma



data = generate_data(100)
plot(data)

p, mu, sigma = EM(data, n_classes=3, iters=0, show=True)

