import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from active_learning import ActiveSRLearner
from active_learning.components.active_criterion import VarianceEnsembleMethod
from active_learning.components.query_strategies import Reject
from active_learning.data import functions
from sklearn.ensemble import ExtraTreesRegressor

# CREATE INITIAL DATASET
fun = functions.grammacy_lee_2009
bounds = np.array(functions.bounds[fun])
n = 50
X_train = pd.DataFrame(
    {'x1': (bounds[0, 0] - bounds[0, 1]) * np.random.rand(n) + bounds[0, 1],
     'x2': (bounds[1, 0] - bounds[1, 1]) * np.random.rand(n) + bounds[1, 1],
     })
y_train = -fun(X_train)

# QUERY NEW POINTS
active_learner = ActiveSRLearner(
    VarianceEnsembleMethod(
        base_ensemble=ExtraTreesRegressor(max_features=0.5, bootstrap=True)),
    Reject(
        bounds, num_eval=int(200)),
    X_train,
    y_train,
    bounds=bounds)

X_new = active_learner.query(40)
print(X_new)

# PLOTS
prediction = active_learner.active_criterion.function
criterion = active_learner.active_criterion

n_plot = 200
x1_plot = np.linspace(bounds[0, 0], bounds[0, 1], n_plot)
x2_plot = np.linspace(bounds[1, 0], bounds[1, 1], n_plot)
mesh = np.meshgrid(x1_plot, x2_plot)
X_plot = pd.DataFrame({'x1': mesh[0].ravel(), 'x2': mesh[1].ravel()})

plt.figure()
plt.pcolormesh(x1_plot, x2_plot, prediction(X_plot).reshape(n_plot, n_plot), cmap="RdBu_r")
plt.colorbar()
plt.scatter(X_train.x1, X_train.x2, label="training samples")
plt.scatter(X_new.x1, X_new.x2, label="new proposed samples")
plt.title("Predictions")
plt.legend()
plt.savefig("examples/example.png")

plt.figure()
plt.pcolormesh(x1_plot, x2_plot, criterion(X_plot).reshape(n_plot, n_plot), cmap="RdBu_r")
plt.colorbar()
plt.scatter(X_train.x1, X_train.x2, label="training samples")
plt.scatter(X_new.x1, X_new.x2, label="new proposed samples")
plt.title("Estimated Error")
plt.legend()
plt.show()
plt.savefig("examples/errors.png")
