import numpy as np
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class RidgeMax:
    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True) -> None:
        """
        Ridge Regression model.

        Parameters
        ----------
        alpha : float, default=1.0
            Regularization strength; must be a non-negative number.
        fit_intercept : bool, default=True
            Whether to calculate the intercept for this model. If False, no intercept is used.
        """
        if not isinstance(alpha, (int, float)):
            raise ValueError("alpha must be numeric.")
        if alpha < 0:
            raise ValueError("alpha must be non-negative.")
        self.alpha: float = alpha
        self.fit_intercept: bool = fit_intercept
        self.coef_: Optional[np.ndarray] = None  # holds the estimated coefficients
        self._n_features: Optional[int] = None   # number of features in the training data

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeMax":
        """
        Fit the ridge regression model to the training data.

        Parameters
        ----------
        X : np.ndarray
            2D array of shape (n_samples, n_features) representing the input features.
        y : np.ndarray
            Target values. Either:
              - 1D array of shape (n_samples,) for a single target, or
              - 2D array of shape (n_samples, n_targets) for multiple targets.

        Returns
        -------
        self : RidgeMax
            Fitted estimator.
        """
        # Validate X
        if not isinstance(X, np.ndarray):
            raise ValueError("X must be a numpy array.")
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        n_samples, n_features = X.shape
        self._n_features = n_features

        # Validate y and adjust shape if necessary
        if not isinstance(y, np.ndarray):
            raise ValueError("y must be a numpy array.")
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        elif y.ndim == 2:
            if y.shape[0] != n_samples:
                raise ValueError("The number of samples in X and y must match.")
        else:
            raise ValueError("y must be a 1D or 2D numpy array.")

        # Build the design matrix (vectorized via np.concatenate)
        if self.fit_intercept:
            # Concatenate a column of ones without an explicit hstack call
            X_design = np.concatenate([X, np.ones((n_samples, 1))], axis=1)
        else:
            X_design = X

        # Compute the Gram matrix A = X_design^T @ X_design
        A = X_design.T @ X_design  # Shape: (n_params, n_params)
        # Instead of constructing a full regularization matrix, add alpha directly:
        if self.fit_intercept:
            # Only regularize the coefficients, not the intercept
            A[:n_features, :n_features] += self.alpha * np.eye(n_features)
        else:
            A += self.alpha * np.eye(n_features)

        # Compute the right-hand side b
        b = X_design.T @ y

        # Solve for beta (ridge regression coefficients)
        try:
            beta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError as e:
            logging.error("Failed to solve the linear system: %s", e)
            raise

        # For a univariate target, return a 1D array
        if beta.shape[1] == 1:
            beta = beta.ravel()
        self.coef_ = beta

        logging.info("Model fitted successfully.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the fitted ridge regression model.

        Parameters
        ----------
        X : np.ndarray
            2D array of shape (n_samples, n_features).

        Returns
        -------
        y_pred : np.ndarray
            Predicted values. For univariate targets, a 1D array is returned.
        """
        if self.coef_ is None:
            raise ValueError("Model is not fitted yet. Please call fit() first.")
        if not isinstance(X, np.ndarray):
            raise ValueError("X must be a numpy array.")
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        n_samples, n_features = X.shape
        if n_features != self._n_features:
            raise ValueError(
                f"Mismatch in number of features: expected {self._n_features}, got {n_features}."
            )

        if self.fit_intercept:
            X_design = np.concatenate([X, np.ones((n_samples, 1))], axis=1)
        else:
            X_design = X

        y_pred = X_design @ self.coef_
        return y_pred

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the coefficient of determination R² of the prediction.

        Parameters
        ----------
        X : np.ndarray
            Test samples.
        y : np.ndarray
            True values for X.

        Returns
        -------
        score : float
            R² score.
        """
        y_pred = self.predict(X)
        # Handle both univariate and multivariate targets
        if y.ndim == 1 or y_pred.ndim == 1:
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return 1 - ss_res / ss_tot
        else:
            ss_res = np.sum((y - y_pred) ** 2, axis=0)
            ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2, axis=0)
            return float(1 - np.mean(ss_res / ss_tot))


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the Mean Squared Error between the true and predicted values.

    Parameters
    ----------
    y_true : np.ndarray
        True target values.
    y_pred : np.ndarray
        Predicted target values.

    Returns
    -------
    mse : float
        Mean Squared Error.
    """
    return float(np.mean((y_true - y_pred) ** 2))


def main() -> None:
    np.random.seed(42)
    n_samples = 200
    n_features = 3
    n_targets = 2  # multivariate

    # true_coef is of shape (n_features, n_targets) and true_intercept is of shape (n_targets,)
    true_coef = np.array([[1.5, -2.0],
                          [-2.0,  3.0],
                          [3.0,  0.5]])
    true_intercept = np.array([0.5, -1.0])

    # Generate synthetic data
    X = np.random.randn(n_samples, n_features)
    noise = np.random.randn(n_samples, n_targets) * 0.5 + np.random.randn(n_samples, n_targets) * 0.7
    y = X @ true_coef + true_intercept + noise

    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    split_index = int(0.8 * n_samples)
    train_idx, test_idx = indices[:split_index], indices[split_index:]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # fit
    alpha = 1.0
    model = RidgeMax(alpha=alpha, fit_intercept=True)
    model.fit(X_train, y_train)

    # preds
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # performance metrics
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    r2_train = model.score(X_train, y_train)
    r2_test = model.score(X_test, y_test)

    # logs
    if model.fit_intercept:
        logging.info("Estimated coefficients (excluding intercept):\n%s", model.coef_[:-1])
        logging.info("Estimated intercept:\n%s", model.coef_[-1])
    else:
        logging.info("Estimated coefficients:\n%s", model.coef_)

    print("\nPerformance Metrics:")
    print(f"Train MSE: {mse_train:.4f}")
    print(f"Test MSE: {mse_test:.4f}")
    print(f"Train R²: {r2_train:.4f}")
    print(f"Test R²: {r2_test:.4f}")


if __name__ == "__main__":
    main()
