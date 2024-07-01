import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

from active_learning import ActiveSRLearner
from active_learning.benchmark import functions
from active_learning.components.active_criterion import ServiceVarianceEnsembleMethod, ServiceVarianceCriterion
from active_learning.components.query_strategies import ServiceQueryVariancePDF, ServiceQueryMax
from sklearn.model_selection import ShuffleSplit

fun = functions.grammacy_lee_2009  # The function we want to learn
bounds = np.array(functions.bounds[fun])  # [x1 bounds, x2 bounds]
n = 50
X_train = pd.DataFrame(
    {'x1': (bounds[0, 0] - bounds[0, 1]) * np.random.rand(n) + bounds[0, 1],
     'x2': (bounds[1, 0] - bounds[1, 1]) * np.random.rand(n) + bounds[1, 1],
     })  # Initiate distribution
y_train = -fun(X_train)


def test_base_functionalities():


    active_criterion = ServiceVarianceEnsembleMethod(  # Parameters to be used to estimate the surface response
        estimator=ExtraTreesRegressor(  # Base estimator for the surface
            max_features=0.8, bootstrap=True)
    )
    query_strategy = ServiceQueryVariancePDF(bounds, num_eval=int(20000))

    # QUERY NEW POINTS
    active_learner = ActiveSRLearner(
        active_criterion,  # Active criterion yields a surface
        query_strategy,  # Given active criterion surface, execute query
        X_train,  # Input data X
        y_train,  # Input data y (target)
        bounds=bounds)

    X_new = active_learner.query(3)
    assert len(X_new) == 3
    assert X_new.shape[1] == X_train.shape[1]

    query_strategy = ServiceQueryMax(x0=bounds.mean(axis=1), bounds=bounds)

    active_learner = ActiveSRLearner(
        active_criterion,
        query_strategy,
        X_train,
        y_train,
        bounds=bounds)

    X_new = active_learner.query()

    assert len(X_new) == 1
    assert X_new.shape[1] == X_train.shape[1]


def test_variance_criterion():
    active_criterion = ServiceVarianceCriterion(ExtraTreesRegressor(), ShuffleSplit())
    query_strategy = ServiceQueryVariancePDF(bounds, num_eval=int(20000))
    active_learner = ActiveSRLearner(
        active_criterion,
        query_strategy,
        X_train,
        y_train,
        bounds=bounds)

    X_new = active_learner.query(3)

    assert len(X_new) == 3
    assert X_new.shape[1] == X_train.shape[1]


def test_variance_criterion_with_maximum():
    active_criterion = ServiceVarianceCriterion(ExtraTreesRegressor(), ShuffleSplit())
    query_strategy = ServiceQueryMax(x0=bounds.mean(axis=1), bounds=bounds)
    active_learner = ActiveSRLearner(
        active_criterion,
        query_strategy,
        X_train,
        y_train,
        bounds=bounds)

    X_new = active_learner.query()

    assert len(X_new) == 1
    assert X_new.shape[1] == X_train.shape[1]
