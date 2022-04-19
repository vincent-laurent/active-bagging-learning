from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from smt.surrogate_models import KRG


class SurrogateKRG(BaseEstimator, RegressorMixin):
    """
    Sklearn api for SMT models
    """

    def __init__(self, theta0: iter = None, print_global: bool = False):
        if theta0 is None:
            theta0 = [1e-2, ]
        self.theta0 = theta0
        self.print_global = print_global

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        self.X_ = X
        self.y_ = y

        self.sm_ = KRG(theta0=self.theta0, print_global=self.print_global)
        self.sm_.set_training_values(X, y)
        self.sm_.train()
        # Return the classifier
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        return self.sm_.predict_values(X)
