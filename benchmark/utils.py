import numpy as np
import pandas as pd
import matplotlib.pyplot as plot


def integrate(f: callable, bounds: iter, num_mc=int(1E6)):
    sampling = np.random.uniform(size=num_mc * len(bounds)).reshape(
        -1, len(bounds)
    )
    for i, (min_, max_) in enumerate(bounds):
        sampling[i] = sampling[i] * (max_ - min_) + min_
    return np.sum(f(sampling)) / len(sampling)


def evaluate(true_function, learned_surface, bounds, num_mc=int(1E6), l=2):
    def f(x):
        return np.abs(np.ravel(true_function(x)) - np.ravel(learned_surface(x))) ** l

    return integrate(f, bounds, num_mc) ** (1/l)


def eval_surf_2d(fun, bounds, num=200):
    xx = np.linspace(bounds[0, 0], bounds[0, 1], num=num)
    yy = np.linspace(bounds[1, 0], bounds[1, 1], num=num)
    x, y = np.meshgrid(xx, yy)
    x = pd.DataFrame(dict(x0=x.ravel(), x1=y.ravel()))
    z = fun(x.values)
    return xx, yy, x, z


if __name__ == '__main__':
    def true_function(x):
        return np.ones_like(x[:, 0])


    def learned_surface(x):
        return 1*np.ones_like(x[:, 0])


    bounds = [[0, 5], [0, 1]]
    a = evaluate(true_function, learned_surface, bounds)
    print(a)
