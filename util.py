import numpy as np

def nmse(y_pred, y_true):
    # """Normalized mean squared error regression loss
    # Parameters
    # ----------
    # y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
    #     Ground truth (correct) target values.
    # y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
    #     Estimated target values.
    # Returns
    # -------
    # loss : float or ndarray of floats
    #     A non-negative floating point value (the best value is 0.0), or an
    #     array of floating point values, one for each individual target.
    # """
    mse = np.square(np.subtract(y_true, y_pred)).mean()
    means = np.ones_like(y_true) * y_true.mean()
    norm = np.square(np.subtract(means, y_true)).mean()
    return mse / norm

def percentage_difference(values):
    num = np.abs(values[0] - values[1])
    den = np.sum(values) / 2
    return (num / den) * 100

# def average(data):
#     return np.mean(data)

# def variance(data):
#     return np.var(data)

# def minimum(data):
#     return np.min(data)

# def maximum(data):
#     return np.max(data)

# def median(data):
#     return np.median(data)