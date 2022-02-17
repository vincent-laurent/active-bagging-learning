import numpy as np
import typing

def check_2d(x: np.ndarray):
    if len(x.shape) != 2 or x.shape[1] == 2:
        raise ValueError("fail to understand input data")


def annie_sauer_2021(x: iter):
    """
    f (x) =
    - 1.35 cos(12πx) x ∈ [0, 0.33]
    - 1.35 x ∈ [0.33, 0.66]
    - 1.35 cos(6πx) x ∈ [0.66, 1].
    """
    x = np.array(x)
    y = np.where(x < 0, np.nan, x)
    y = np.where(x < 0.33, 1.35 * np.cos(12 * np.pi * x), y)
    y = np.where(x < 0.66, 1.35, y)
    y = np.where(x <= 1, 1.35 * np.cos(6 * np.pi * x), y)
    y = np.where(x > 1, np.nan, y)
    return y


def grammacy_lee_2009(x: typing.Iterable[typing.Iterable]):
    """ (x1, x2) = 10 x_1 exp (−x^2_1 − x^2_2) for x1, x2 ∈ [−2, 4]"""

    x = np.array(x)

    check_2d(x)

    x_square = x ** 2
    return 10 * x[:, 0] * np.exp(-x_square[:, 0] - x_square[:, 1])


def himmelblau(x: typing.Iterable[typing.Iterable]):
    x = np.array(x)

    check_2d(x)

    return (x[:, 0] ** 2 + x[:, 1] - 11) ** 2 + (
            x[:, 0] + x[:, 1] ** 2 - 7) ** 2


def branin(x: typing.Iterable[typing.Iterable],
           a=1, b=5.1 / (4 * pow(np.pi, 2)), c=5 / np.pi, r=6,
           s=10, t=1 / (8 * np.pi)):
    x = np.array(x)
    check_2d(x)
    return a * (x[:, 1] - b * x[:, 0] ** 2 + c * x[:, 0] - r) ** 2 + s * (
            1 - t) * np.cos(x[:, 0]) + s
