#!/usr/bin/env python
# coding: utf-8
"""
ACTIVE LEARNING FOR A SIMPLE 1D FUNCTION

This example is a simple application of active learning.
The 1D function allows for easy vizualisation of the model and its error
estimate.
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import BaggingRegressor

from active_learning.components.utils import get_variance_function
from active_learning.components.query_strategies import random_sampling_in_domain

SEED = 1234
RNG = np.random.default_rng(seed=SEED)
matplotlib.use("Qt5Agg")


def unknown_function(x):
    return x ** 5 * np.sin(10 * np.pi * x)


bounds = (np.array([0.0]), np.array([1.0]))
nb_initial_samples = 5

x = RNG.uniform(low=0, high=1, size=(nb_initial_samples, 1))
y = unknown_function(x)
training_dataset = np.hstack([x, y])  # In this example, the training examples are stored in a numpy array.

domain = np.linspace(0, 1, 100)  # Just for plotting the prediction function

nb_iter = 16
fig, axs = plt.subplots(4, 4, sharey=True, sharex=True)

for iter, ax in enumerate(axs.ravel()):

    # TRAINING
    model = BaggingRegressor(n_estimators=20, random_state=SEED)
    model.fit(training_dataset[:, [0]], training_dataset[:, 1])

    # PLOT
    prediction = model.predict(domain[:, np.newaxis])
    error = get_variance_function(model.estimators_)(domain.reshape(-1, 1))
    ax.plot(domain, prediction, color="C1")
    ax.plot(domain, unknown_function(domain), color="grey", linestyle="--")
    ax.fill_between(domain.ravel(), prediction - error / 2, prediction + error / 2, color="C1", alpha=0.5)
    ax.scatter(training_dataset[:, 0], training_dataset[:, 1], color="C0")
    if iter > 0: ax.scatter(new_samples, unknown_function(new_samples), color="C2")
    ax.set_title("iter={}".format(iter))
    ax.set_ylim(-1, 1)

    # ADDING NEW SAMPLES TO THE DATABASE
    new_samples = random_sampling_in_domain(get_variance_function(model.estimators_), bounds=bounds, nb_samples=3,
                                            rng=RNG)
    new_outputs = unknown_function(new_samples)
    training_dataset = np.vstack([training_dataset, np.hstack([new_samples, new_outputs])])

plt.show()
