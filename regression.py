# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def load_data():
    file_name = 'data.csv'
    # file_name = 'data2.csv'
    data = np.loadtxt(file_name, delimiter=',')
    x = data[:, 0]
    y = data[:, 1]
    return x, y


def plot(x, y, w=None):
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, color='r', edgecolor='r', s=80, label="Samples")

    # also plot the prediction
    if not w is None:
        x_plot = np.linspace(-2, 6, 100)
        m = x_plot.shape[0]
        X_plot = np.vstack([x_plot, np.ones(m)]).T
        plt.plot(x_plot, np.dot(X_plot, w), linewidth=5, color='tab:blue', label="Model")

    plt.show()


def regression(x, y):

    ### Please fill in here!

    w = None
    return w

x, y = load_data()
plot(x, y)
w = regression(x, y)
plot(x, y, w)
