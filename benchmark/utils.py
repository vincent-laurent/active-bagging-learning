import numpy as np


def integrate(f: callable, bounds: iter, num_mc=int(1E6)):
    sampling = np.random.uniform(size=num_mc * len(bounds)).reshape(
        -1, len(bounds)
    )
    for i, (min_, max_) in enumerate(bounds):
        sampling[i] = sampling[i] * (max_ - min_) + min_
    return np.sum(f(sampling)) / len(sampling)


def evaluate(true_function, learned_surface, bounds, num_mc=int(1E6)):
    def error(x):
        return (true_function(x) - learned_surface(x)) ** 2

    return np.sqrt(integrate(error, bounds, num_mc))


if __name__ == '__main__':
    def true_function(x):
        return x[:, 1] ** 2


    def learned_surface(x):
        return x[:, 1] ** 2.1


    bounds = [[0, 1], [0, 1]]
    a = evaluate(true_function, learned_surface, bounds)
    print(a)
