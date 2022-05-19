from base import ActiveSRLearner
import matplotlib.pyplot as plot
import numpy as np


class Analysis:
    def __init__(self, learner: ActiveSRLearner):
        self.learner = learner
        bounds = self.learner.bounds
        self.xx = np.linspace(bounds[0, 0], bounds[0, 1], num=200)
        self.yy = np.linspace(bounds[1, 0], bounds[1, 1], num=200)

    def animate_surf(self, ax:plot.Axes):
        from matplotlib import animation
        ax.plot(data.year[:i], data.gdpPercap[:i], color)


if __name__ == '__main__':
    analysis = Analysis(a)
    self = analysis