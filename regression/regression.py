# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def load_data(file_name):
    data = np.loadtxt(file_name, delimiter=',')
    x = data[:, 0]
    y = data[:, 1]
    return x, y


def plot(x, y, w=None):
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, color='r', edgecolor='r', s=80, label="Samples")

    # also plot the prediction
    if not w is None:
        deg = w.shape[0]
        x_plot = np.linspace(-2, 6, 100)
        X_plot = np.vander(x_plot, deg)
        plt.plot(x_plot, np.dot(X_plot, w), linewidth=5, color='tab:blue', label="Model")

    plt.show()


def regression(x, y, deg=3):
    X = np.vander(x, deg+1)
    w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y))
    return w

x, y = load_data('data.csv')
w = regression(x, y, deg=1)
plot(x, y, w)

x, y = load_data('data2.csv')
w = regression(x, y, deg=3)
plot(x, y, w)
