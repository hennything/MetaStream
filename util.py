import numpy as np

def nmse(y_pred, y_true):
    mse = np.square(np.subtract(y_true, y_pred)).mean()
    means = np.ones_like(y_true) * y_true.mean()
    norm = np.square(np.subtract(means, y_true)).mean()
    return mse / norm

def percentage_difference(values):
    num = np.abs(values[0] - values[1])
    den = np.sum(values) / 2
    return (num / den)