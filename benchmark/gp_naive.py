import numpy as np
import base
from data import functions
from benchmark.utils import evaluate
from sampling import latin_square

functions_ = list(functions.bounds.keys())

fun = functions_[1]

def run(fun):
    bounds = functions.bounds[fun]
    x0 = latin_square.iterative_sampler(x_limits=np.array(bounds))
    active_learner = base.ActiveSRLearner(
        base.gaussian_est,
        base.iterative_sampler,
        x0, fun(x0)
    )
    x_new = active_learner.run(10)