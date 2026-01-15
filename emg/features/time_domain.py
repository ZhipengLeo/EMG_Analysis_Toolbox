import numpy as np


def rms(x: np.ndarray) -> float:
    return np.sqrt(np.mean(x ** 2))


def mav(x: np.ndarray) -> float:
    return np.mean(np.abs(x))


def var(x: np.ndarray) -> float:
    return np.var(x, ddof=1)


def ssc(x: np.ndarray, threshold: float = 5e-6) -> int:
    dx1 = x[1:-1] - x[:-2]
    dx2 = x[1:-1] - x[2:]
    return np.sum((dx1 * dx2 > threshold))


def aac(x: np.ndarray) -> float:
    return np.mean(np.abs(np.diff(x)))
