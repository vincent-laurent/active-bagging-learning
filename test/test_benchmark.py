import numpy as np

from active_learning.benchmark import functions

x2d = np.array([0, 0]).reshape(1, -1)


def test_call_benchmark_function():
    functions.grammacy_lee_2009(x2d)
    functions.marelli_2018(x2d)
    functions.grammacy_lee_2009_rand(x2d)
    functions.annie_sauer_2021(x2d)
    functions.branin(x2d)
    functions.branin_rand(x2d)
    functions.himmelblau(x2d)
    functions.himmelblau_rand(x2d)
    functions.golden_price(x2d)
    functions.golden_price_rand(x2d)
    functions.synthetic_2d_1(x2d)
    functions.synthetic_2d_2(x2d)


def test_plot_benchamrk():
    functions.plot_benchamrk_functions()