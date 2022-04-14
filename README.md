
# Active  Strategy for surface response estimation
###### tags: `Surrogate Model` `Machine Learning`
**Literature :**
* **Review** [Simpson2001](https://ntrs.nasa.gov/api/citations/19990087092/downloads/19990087092.pdf) 
![](https://i.imgur.com/w571mZ7.png)
* **Reliability** in [[Marelli2018]](https://arxiv.org/pdf/1709.01589) using polynomial chaos expansion. The problem is to find a region defined by a function $\{x ; \, g(x) \leqslant 0\}$ where $g$ is called limit state function. *Bootstrap approach to estimate variance* 
* **Properties in multilayer percpetron network** [[Fukumizu2000]](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.51.1885&rep=rep1&type=pdf) regression problem. Active learning : resampling trapped in local minima ? Redundancy of hidden units in active learning
* Gaussian process using mutual information 
* **Surface response methodology** [[Bezerra2008]](https://d1wqtxts1xzle7.cloudfront.net/45518928/Response_Surface_Methodology_RSM_as_a_20160510-11788-z5s7f4-with-cover-page-v2.pdf?Expires=1647600354&Signature=FWuGdH4xQIPYbo6gjfofYOvSiNCZknuwktVpgOuRU0wbBAjHhrN2a2cYCoLaqFmhLzuJNl~TeX2iXFh7rYFlAfgBwqQh6-lV29XxuU6AJTqj6lkP2MaIMHke4RMcJ6mJN39lXcfg6Ohf5D9TnD7v-Eze4fHCHbklEk9REPok6O0V3MIvx7A4XriV5Tffe5yu1HZ1fCuHBULS5PiRyuRBzKavclvPFQBPDWx5-J~y9a85oB6JGcey3VId7fvtfRUGXXn49WqHm3fJfqpLbYj62drFGjE6XcmBWm1CzBn0Guaf~ig8k6JfI9wOrErxofAkR8tjnd51VUAelB0XCY4v1A__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA) based on linear models
## Context 

Plug in approach to active learning for surface response estimation

* The objective is to approximate function $f \in \mathcal{X} \rightarrow \mathbb{R}^n$.
* **Objective :** find an estimation of $f$, $\hat{f}$ in a family of measurable function $\mathcal{F}$ such that $$ f^* = \underset{\hat{f} \in \mathcal{F}}{\text{argmin}} \|f - \hat{f} \| $$ 
* At time $t$ we dispose of a set of $n$ evaluations $$(x_i, f(x_i))_{i\leqslant n}$$ 
* All feasible points can be sampled, we assume having a implicitly defined domain $\mathcal{X}$

## Usage

```python 

import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit

import base
import components.query_strategies as qs
from components import active_criterion
from data import functions
from models.smt_api import SurrogateKRG
from sampling import latin_square


name = "grammacy_lee_2009_rand"
fun = functions.__dict__[name]          # The function we want to learn
estimator = dict(                       # Parameters to be used to estimate the surface response
    base_estimator=SurrogateKRG(),      # Base estimator for the surface
    splitter=ShuffleSplit(n_splits=2))  # Resampling strategy
bounds = [[0, 1], [0, 1]]               # [x bounds, y bounds]
x0 = latin_square.iterative_sampler(
    x_limits=np.array(bounds), size=n0,
    batch_size=n0 // 2)

active_learner = base.ActiveSRLearner(
    active_criterion.estimate_variance,     # Active criterion
    qs.reject_on_bounds,                    # Given active crietrion, 
    pd.DataFrame(x0),                       # Input data X
    pd.DataFrame(fun(x0)),                  # Input data y (target)
    bounds=np.array(bounds),                # Bounds
    estimator_parameters=estimator)         
x_new = active_learner.query(1)             # Request one point

```

To use the script, one have to dispose of

* Surface response estimator (linear model, gaussian vectors, etc.) in api sklearn
* A resampling 