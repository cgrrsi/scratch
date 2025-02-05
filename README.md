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

```
from olsmax import OLSMax
# fit
model = OLSMax(fit_intercept=True)
model.fit(X_train, y_train, method='pinv')
# pred
y_pred = model.predict(X_test)
# betas
betas = model.get_params()
```




