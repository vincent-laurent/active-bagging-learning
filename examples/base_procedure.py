import numpy as np
import pandas as pd

from sklearn.model_selection import ShuffleSplit
from models.smt_api import SurrogateKRG

from active_learning import ActiveSRLearner
from active_learning.data import functions
from active_learning.components.active_criterion import estimate_variance
from active_learning.components.query_strategies import reject_on_bounds

fun = functions.grammacy_lee_2009
bounds = np.array(functions.bounds[fun])

X = pd.DataFrame(np.random.rand(20, 2), columns=['x1', 'x2'])
y = -fun(X)
active_learner = ActiveSRLearner(estimate_variance, reject_on_bounds, X, y,
                                 bounds=bounds,
                                 estimator_parameters=dict(
                                     base_estimator=SurrogateKRG(),
                                     splitter=ShuffleSplit(n_splits=5)
                                 ),
                                 query_parameters=dict(
                                     batch_size=40,  # Does a 40 points LHS
                                 ))
x_new = active_learner.query(2)
print(x_new)
