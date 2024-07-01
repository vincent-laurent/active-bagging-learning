import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

from active_learning import ActiveSRLearner
from active_learning.benchmark import functions
from active_learning.components.active_criterion import ServiceVarianceEnsembleMethod
from active_learning.components.query_strategies import ServiceQueryVariancePDF

plt.style.use("bmh")
plt.rcParams["font.family"] = "ubuntu"
plt.rcParams['axes.facecolor'] = "white"

# CREATE INITIAL DATASET
fun = functions.grammacy_lee_2009
bounds = np.array(functions.bounds[fun])
n = 50
X_train = pd.DataFrame(
    {'x1': (bounds[0, 0] - bounds[0, 1]) * np.random.rand(n) + bounds[0, 1],
     'x2': (bounds[1, 0] - bounds[1, 1]) * np.random.rand(n) + bounds[1, 1],
     })
y_train = -fun(X_train)

active_criterion = ServiceVarianceEnsembleMethod(
    estimator=ExtraTreesRegressor(max_features=0.8, bootstrap=True))
query_strategy = ServiceQueryVariancePDF(bounds, num_eval=int(20000))

# QUERY NEW POINTS
active_learner = ActiveSRLearner(
    active_criterion,
    query_strategy,
    X_train,
    y_train,
    bounds=bounds)

X_new = active_learner.query(3)

# PLOTS
prediction = active_learner.active_criterion.function
criterion = active_learner.active_criterion

n_plot = 200
x1_plot = np.linspace(bounds[0, 0], bounds[0, 1], n_plot)
x2_plot = np.linspace(bounds[1, 0], bounds[1, 1], n_plot)
mesh = np.meshgrid(x1_plot, x2_plot)
X_plot = pd.DataFrame({'x1': mesh[0].ravel(), 'x2': mesh[1].ravel()})

filled_marker_style = dict(marker='o', lw=0, markersize=5,
                           color='k',
                           markerfacecolor="#4eca5b",
                           markerfacecoloralt='lightsteelblue',
                           markeredgecolor='darkgreen')

plt.figure(figsize=(7, 6), dpi=100)
plt.pcolormesh(x1_plot, x2_plot, prediction(X_plot).reshape(n_plot, n_plot), cmap="RdBu_r")
plt.colorbar()
plt.scatter(X_train.x1, X_train.x2, label="training samples", c="k", marker=".")
plt.plot(X_new.x1, X_new.x2, label="new proposed samples", **filled_marker_style)
plt.title("Predictions")
plt.legend()
plt.savefig("public/example.png")

plt.figure(figsize=(7, 6), dpi=100)
plt.pcolormesh(x1_plot, x2_plot, criterion(X_plot).reshape(n_plot, n_plot), cmap="RdBu_r")
plt.colorbar()
plt.scatter(X_train.x1, X_train.x2, label="training samples", c="k", marker=".")
plt.plot(X_new.x1, X_new.x2, label="new proposed samples", **filled_marker_style)
plt.title("Active surface")
plt.legend()
plt.show()
plt.savefig("public/active_surface.png")
