import typing

import numpy as np


def check_2d(x: np.ndarray):
    if len(x.shape) != 2 or x.shape[1] != 2:
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

    x_ = np.array(x)

    check_2d(x)

    x_square = x_ ** 2
    return 10 * x_[:, 0] * np.exp(-x_square[:, 0] - x_square[:, 1])


def himmelblau(x: typing.Iterable[typing.Iterable]):
    x_ = np.array(x)

    check_2d(x_)

    return (x_[:, 0] ** 2 + x_[:, 1] - 11) ** 2 + (
            x_[:, 0] + x_[:, 1] ** 2 - 7) ** 2


def branin(x: typing.Iterable[typing.Iterable],
           a=1, b=5.1 / (4 * pow(np.pi, 2)), c=5 / np.pi, r=6,
           s=10, t=1 / (8 * np.pi)):
    x_ = np.array(x)
    check_2d(x_)
    return a * (x_[:, 1] - b * x_[:, 0] ** 2 + c * x_[:, 0] - r) ** 2 + s * (
            1 - t) * np.cos(x_[:, 0]) + s


def golden_price(x):
    x_ = np.array(x)
    check_2d(x)
    xx, yy = x_[:, 0], x_[:, 1]
    return (1 + (xx + yy + 1) ** 2 * (
            19 - 14 * xx + 3 * xx ** 2 - 14 * yy + 6 * xx * yy + 3 * yy ** 2)) * (
                   30 + (2 * xx - 3 * yy) ** 2 * (
                   18 - 32 * xx + 12 * xx ** 2 + 48 * yy - 36 * xx * yy + 27 * yy ** 2))


bounds = {
    annie_sauer_2021: [[0, 1]],
    grammacy_lee_2009: [[-4, 4], [-4, 4]],
    golden_price: [[0, 1], [0, 1]],
    branin: [[0, 1], [0, 1]],
    himmelblau: [[-5, 5], [-5, 5]],
}
