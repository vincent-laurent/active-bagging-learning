from sampling.latin_square import iterative_sampler


def reject_on_bounds(X, y, coverage_function, size=10, batch_size=50,
                     bounds=None):
    from scipy.stats import rankdata
    if bounds is None:
        x_new = iterative_sampler(X, size=batch_size)
    else:
        x_new = iterative_sampler(x_limits=bounds, size=batch_size)
    cov = coverage_function(x_new)
    order = rankdata(-cov, method="ordinal")
    selector = order <= size
    return x_new[selector], cov[selector]


def reject_on_bounds_ada(X, y, coverage_function, size=10,
                         bounds=None, alpha=2):
    batch_size = min(int(len(X) * alpha), 5 * size)
    from scipy.stats import rankdata
    if bounds is None:
        x_new = iterative_sampler(X, size=batch_size)
    else:
        x_new = iterative_sampler(x_limits=bounds, size=batch_size)
    cov = coverage_function(x_new)
    order = rankdata(-cov, method="ordinal")
    selector = order <= size
    return x_new[selector], cov[selector]
