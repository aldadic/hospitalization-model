import numpy as np


def mape(actual, pred):
    return np.mean(np.abs(actual - pred) / actual) * 100


def mae(actual, pred):
    return np.mean(np.abs(actual - pred))


def mase(actual, pred):
    numerator = np.mean(np.abs(actual - pred))
    denominator = np.abs(np.diff(actual)).sum() / (len(actual) - 1)
    return numerator / denominator
