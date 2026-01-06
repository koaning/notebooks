# TIL: Warm Starts Only Matter for Iterative Solvers

When sweeping regularization parameters in scikit-learn, warm starts can speed things up — but only for models that use iterative optimization.

## Ridge Regression: Warm Start is Irrelevant

Ridge has a closed-form solution. Using SVD, you can compute coefficients for **all** alpha values almost as fast as one:

```python
U, s, Vt = np.linalg.svd(X, full_matrices=False)
UTy = U.T @ y
# For ANY alpha, just do:
d = s / (s**2 + alpha)
coefs = (d * UTy) @ Vt
```

This is why `RidgeCV` is so fast and why `Ridge` doesn't have a `warm_start` parameter.

## Logistic Regression: Warm Start Helps a Lot

Logistic Regression uses iterative solvers (L-BFGS, SAG, etc.). When sweeping C values, adjacent solutions are similar — making the previous solution a great starting point.

```python
model = LogisticRegression(warm_start=True, solver='lbfgs')
for C in np.logspace(-3, 2, 50):
    model.C = C
    model.fit(X, y)  # starts from previous coefs
```

Typical speedup: **2-3x** over cold starts.

## The Rule

| Model | Closed-form? | Warm start helps? |
|-------|-------------|-------------------|
| Ridge | Yes (SVD) | No — use `RidgeCV` |
| Logistic | No | Yes |
| Lasso/ElasticNet | No | Yes — or use `lasso_path()` |
