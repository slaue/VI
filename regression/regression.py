# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def load_data(file_name, m=None):
    data = np.loadtxt(file_name, delimiter=',')
    x = data[:, 0]
    y = data[:, 1]

    # subsample data
    if not m is None:
        idx = np.random.choice(x.shape[0], m)
        x = x[idx]
        y = y[idx]
    return x, y


def plot(x, y, w=None, sigma=None):
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, color='r', edgecolor='r', s=80, label="Samples")

    # also plot the prediction
    if not w is None:
        deg = w.shape[0]
        x_plot = np.linspace(np.min(x), np.max(x), 100)
        X_plot = np.vander(x_plot, deg)

        # set plotting range properly
        plt.ylim((np.min(y)*1.2, np.max(y)*1.2))

        plt.plot(x_plot, np.dot(X_plot, w), linewidth=5, color='tab:blue', label="Model")

        # also plot confidence intervall if given
        if not sigma is None:
            plt.plot(x_plot, np.dot(X_plot, w)+sigma, linewidth=2, color='tab:cyan')
            plt.plot(x_plot, np.dot(X_plot, w)-sigma, linewidth=2, color='tab:cyan')

    plt.show()


def regression(x, y, deg=3):
    X = np.vander(x, deg+1)
    w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y))
    return w

np.random.seed(0)

x, y = load_data('data.csv')
w = regression(x, y, deg=1)
plot(x, y, w)

x, y = load_data('data2.csv')
w = regression(x, y, deg=3)
plot(x, y, w)

# new problem, please uncomment below
#x, y = load_data('data3.csv')
#w = regression(x, y, deg=9)
#plot(x, y, w)

