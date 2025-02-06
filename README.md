# scratch

Building classical libraries from scratch to learn and improve performances.

## 1°) OLSMax

### Overview
`OLSMax` is a flexible and efficient implementation of Ordinary Least Squares (OLS) regression supporting both univariate and multivariate targets. It allows users to select different closed-form solutions for fitting the model:

- **`pinv`**: Moore-Penrose pseudoinverse
- **`qr`**: QR decomposition
- **`svd`**: Singular Value Decomposition (SVD)

The implementation offers a competitive performance compared to `scikit-learn` and `statsmodels` implementations.

### Usage 

```python
from olsmax import OLSMax

# fit
model = OLSMax(fit_intercept=True)
model.fit(X_train, y_train, method='pinv')

# pred
y_pred = model.predict(X_test)

# betas
betas = model.get_params()
```

### Comparison with other implementations (over 100,000 runs)

| Feature        | OLSMax (pinv) | OLSMax (qr) | OLSMax (svd) | scikit-learn | statsmodels |
|---------------|----------------|---------------|---------------|-------------|-------------|
| Multi-output  | ✅              | ✅             | ✅             | ✅          | ❌          |
| Intercept     | ✅              | ✅             | ✅             | ✅          | ✅          |
| Multiple Solvers | ✅          | ✅             | ✅             | ❌          | ❌          |
| Performance   | Fast      | Fast       | 🔥 Fastest       | Slowest    | Moderate   |
| Robustness    | High      | Lowest     | 🔥 Highest       | Moderate   | High       |

## 2°) RidgeMax

### Overview
`RidgeMax` is an implementation of Ridge Regression built from scratch. It supports both univariate and multivariate targets, automatically handles intercept inclusion, and uses a vectorized closed-form solution. In particular, instead of constructing an entire regularization matrix, it adds the regularization parameter directly to the coefficient sub-block of the Gram matrix.

### Usage

```python
from ridgemax import RidgeMax

# fit
model = RidgeMax(alpha=1.0, fit_intercept=True)
model.fit(X_train, y_train)

# pred
y_pred = model.predict(X_test)

# betas
betas = model.coef_
```
