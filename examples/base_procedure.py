import numpy as np
import pandas as pd

from sklearn.model_selection import ShuffleSplit

from active_learning import ActiveSRLearner
from active_learning.data import functions
from active_learning.models.smt_api import SurrogateKRG
from active_learning.components.active_criterion import estimate_variance
from active_learning.components.query_strategies import reject_on_bounds

# CREATE INITIAL DATASET
fun = functions.grammacy_lee_2009
bounds = np.array(functions.bounds[fun])

X_train = pd.DataFrame(
        {'x1': (bounds[0, 0] - bounds[0, 1]) * np.random.rand(50) + bounds[0, 1],
         'x2': (bounds[1, 0] - bounds[1, 1]) * np.random.rand(50) + bounds[1, 1],
        })
y_train = -fun(X_train)


# QUERY NEW POINTS
active_learner = ActiveSRLearner(estimate_variance, reject_on_bounds, X_train, y_train,
                                 bounds=bounds,
                                 estimator_parameters=dict(
                                     base_estimator=SurrogateKRG(),
                                     splitter=ShuffleSplit(n_splits=5)
                                 ),
                                 query_parameters=dict(
                                     batch_size=40,  # Does a 40 points LHS
                                 ))
X_new = active_learner.query(5)
print(X_new)


# PLOTS
prediction = active_learner.surface
coverage = active_learner.active_criterion

n_plot = 200
x1_plot = np.linspace(bounds[0, 0], bounds[0, 1], n_plot)
x2_plot = np.linspace(bounds[1, 0], bounds[1, 1], n_plot)
mesh = np.meshgrid(x1_plot, x2_plot)
X_plot = pd.DataFrame({'x1': mesh[0].ravel(), 'x2': mesh[1].ravel()})

import matplotlib.pyplot as plt
plt.figure()
plt.pcolormesh(x1_plot, x2_plot, prediction(X_plot).reshape(n_plot, n_plot), cmap="viridis")
plt.colorbar()
plt.scatter(X_train.x1, X_train.x2, label="training samples")
plt.scatter(X_new.x1, X_new.x2, label="new proposed samples")
plt.title("Predictions")
plt.legend()

plt.figure()
plt.pcolormesh(x1_plot, x2_plot, coverage(X_plot).reshape(n_plot, n_plot), cmap="RdBu_r")
plt.colorbar()
plt.scatter(X_train.x1, X_train.x2, label="training samples")
plt.scatter(X_new.x1, X_new.x2, label="new proposed samples")
plt.title("Estimated Error")
plt.legend()

plt.show()
