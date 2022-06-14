#!/usr/bin/env python
# coding: utf-8
"""
ACTIVE LEARNING IN A PRE-EXISTING DATASET

In this example, a database of simulations has been computed preliminary.
A small subset of these simulations is used to train a model, then the
estimated error on the model is used to chose in the existing database new
simulations for the training.

This example can be used to check the efficiency of an active learning strategy
using only already computed results, without any new costly simulations.
"""

import numpy as np
import pandas as pd

from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split

from active_learning.data import functions
from active_learning.components.active_criterion import get_variance_function
from active_learning.components.query_strategies import indices_of_random_sampling_in_finite_set


SEED = 0
RNG = np.random.default_rng(seed=SEED)


def precompute_dataset(nb_samples):
    function = functions.grammacy_lee_2009
    bounds = np.array(functions.bounds[function])

    dataset = pd.DataFrame({
        'x1': RNG.uniform(low=bounds[0, 0], high=bounds[0, 1], size=nb_samples),
        'x2': RNG.uniform(low=bounds[1, 0], high=bounds[1, 1], size=nb_samples),
        })
    dataset["y"] = function(dataset)
    return dataset


def active_learning_training_dataset(full_dataset, nb_samples, *, nb_initial_samples=20, nb_samples_per_iteration=5):
    """Extract `nb_samples` from the `full_dataset` using active learning"""
    initial_indices = RNG.choice(full_dataset.index, size=nb_initial_samples)
    train_dataset = full_dataset.loc[initial_indices]
    remaining_candidates = full_dataset.drop(initial_indices)

    while len(train_dataset) < nb_samples:
        model = BaggingRegressor(random_state=SEED)
        model.fit(train_dataset[["x1", "x2"]], train_dataset["y"])

        def error_estimation(x):
            # Defining a function here just for the `x.values` below. Scikit-learn 1.0.2 returns noisy warnings otherwise.
            return get_variance_function(model.estimators_)(x.values)

        newly_chosen_indices = indices_of_random_sampling_in_finite_set(error_estimation, remaining_candidates[["x1", "x2"]], nb_samples_per_iteration, rng=RNG)
        train_dataset = pd.concat([train_dataset, remaining_candidates.loc[newly_chosen_indices]], axis='index')
        remaining_candidates = remaining_candidates.drop(newly_chosen_indices)

    return train_dataset



full_dataset = precompute_dataset(nb_samples=500)
full_training_dataset, test_dataset = train_test_split(full_dataset, train_size=0.8, random_state=SEED)

# WITH ACTIVE LEARNING
train_dataset_0 = active_learning_training_dataset(full_training_dataset, 100)
model_0 = BaggingRegressor(random_state=SEED)
model_0.fit(train_dataset_0[["x1", "x2"]], train_dataset_0["y"])
print("Model score with active learning:", model_0.score(test_dataset[["x1", "x2"]], test_dataset["y"]))

# WITHOUT ACTIVE LEARNING
random_indices = RNG.choice(full_training_dataset.index, size=100)
train_dataset_1 = full_training_dataset.loc[random_indices]
model_1 = BaggingRegressor(random_state=SEED)
model_1.fit(train_dataset_1[["x1", "x2"]], train_dataset_1["y"])
print("Model score without active learning:", model_1.score(test_dataset[["x1", "x2"]], test_dataset["y"]))

