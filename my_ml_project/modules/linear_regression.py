import numpy as np

class myLinearRegression1D:
    """
    A simple 1D linear regression model using the closed-form solution.

    The model is: y = beta0 + beta1 * x.

    The parameters are estimated as:
      beta1 = sum((x_i - \hat{x})(y_i - \hat{y}) / sum((x_i - \hat{x})^2)
      beta0 = \hat{y} - beta1 * \hat{x},

    where:
      \hat{x}= (1/n) * sum(x_i) and \hat{y}= (1/n) * sum(y_i)
    """

    def __init__(self):
        self.beta0 = None
        self.beta1 = None

    def fit(self, X, y):
        """
        Fit the linear regression model using the direct formula.

        Parameters:
            X (array-like): 1D array or column vector of input features.
            y (array-like): 1D array or column vector of target values.
        """
        # Ensure X and y are 1D arrays
        X = np.ravel(X)
        y = np.ravel(y)

        n = len(X)
        x_bar = np.mean(X)
        y_bar = np.mean(y)

        # Compute the slope (beta1)
        beta1_num = np.sum((X - x_bar) * (y - y_bar))
        beta1_den = np.sum((X - x_bar)**2)
        self.beta1 = beta1_num / beta1_den

        # Compute the intercept (beta0)
        self.beta0 = y_bar - self.beta1 * x_bar

    def predict(self, X):
        """
        Predict target values using the fitted model.

        Parameters:
            X (array-like): 1D array or column vector of input features.

        Returns:
            np.ndarray: Predicted values.
        """
        X = np.ravel(X)
        return self.beta0 + self.beta1 * X

    def get_params(self, deep=True):
        # Since there are no hyperparameters, return an empty dict.
        # If you had hyperparameters, you'd return them in a dict.
        return {}

    def set_params(self, **params):
        # Update any hyperparameters, if present.
        for key, value in params.items():
            setattr(self, key, value)
        return self
