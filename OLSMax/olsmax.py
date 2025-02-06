import numpy as np

class OLSMax:
    """
    A flexible OLS model that allows both single-output (univariate) and multi-output (multivariate) regression and using different closed-form solutions for the fit.
      - 'pinv'  (Moore-Penrose pseudo-inverse)
      - 'qr'    (QR decomposition)
      - 'svd'   (Singular Value Decomposition)
    """
    def __init__(self, fit_intercept=True):
        """
        Args:
            fit_intercept (bool): Whether to include an intercept term in the model.
        """
        self.fit_intercept = fit_intercept
        self.beta_ = None  # shape (n_features [+1], n_outputs)
    
    def _add_intercept(self, X):
        """
        If fit_intercept=True, add a column of 1s to X.
        """
        intercept_col = np.ones((X.shape[0], 1), dtype=X.dtype)
        return np.hstack([intercept_col, X])

    def fit(self, X, y, method="pinv"):
        """
        Fit the model using a closed-form approach.

        Args:
            X (np.ndarray): shape (n_samples, n_features)
            y (np.ndarray): shape (n_samples,) or (n_samples, n_outputs)
            method (str): {'pinv', 'qr', 'svd'}

        Returns:
            self
        """
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        if self.fit_intercept:
            X = self._add_intercept(X)

        if method == "pinv":
            # Beta = pinv(X) @ y
            X_pinv = np.linalg.pinv(X)
            self.beta_ = X_pinv @ y

        elif method == "qr":
            # X = Q R => Beta = R^{-1} Q^T y
            Q, R = np.linalg.qr(X)
            self.beta_ = np.linalg.inv(R) @ (Q.T @ y)

        elif method == "svd":
            # X = U S V^T => Beta = V S^+ U^T y
            U, S, Vt = np.linalg.svd(X, full_matrices=False)
            S_inv = np.diag(1.0 / S)
            self.beta_ = (Vt.T @ S_inv) @ (U.T @ y)

        else:
            raise ValueError("Invalid method. Choose from {'pinv','qr','svd'}.")

        return self

    def predict(self, X):
        """
        Predict with the learned parameters.

        Args:
            X (np.ndarray): shape (n_samples, n_features)

        Returns:
            np.ndarray: shape (n_samples,) or (n_samples, n_outputs)
        """
        if self.beta_ is None:
            raise ValueError("Model not fitted yet.")

        if self.fit_intercept:
            X = self._add_intercept(X)

        y_pred = X @ self.beta_

        # Flatten if single-output
        if y_pred.shape[1] == 1:
            return y_pred.ravel()
        return y_pred

    def get_params(self):
        """
        Returns the learned coefficients (including intercept if any).
        """
        if self.beta_ is None:
            raise ValueError("Model not fitted yet.")
        return self.beta_
