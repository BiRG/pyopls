# pyopls
##### Orthogonal Projection to Latent Structures in Python. 

This pacakge provides a scikit-learn-style regressor to perform OPLS(-DA). OPLS is a pre-processing method to remove
variation from the descriptor variables that are not correlated to the target variable (1).

A 1-component PLS regression is performed to evaluate the filtering.


This implementation is partially based on [Paul Anderson's MATLAB implementation](https://github.com/Anderson-Lab/OPLS).
## Installation
Install from this repository using `pip`:
```
pip install git+git://github.com/BiRG/pyopls.git
```
This package requires Python 3.5+
## OPLS Estimator
### Fit a regressor
Calling `OPLS.fit()` will calculate the orthogonal components and train a one-component PLS model to evaluate the 
filtered data. Other methods trained on the results of `OPLS.get_nonorthogonal()` may perform better.
#### Example
```pythonstub
from pyopls import OPLS
estimator = OPLS(5, scale=True)
estimator.fit(X_train, y_train)
y_pred = estimator.predict(X_test)
```
### Filter orthogonal components from `X`
This can be used as a pre-processing step for other methods.
#### Example
```pythonstub
from sklearn.cross_decomposition import PLSRegression
X_res = estimator.get_nonorthogonal(X)
pls_regressor = PLSRegression(5)
pls_regressor.fit(X_train, y_train)
y_pred = pls_regressor.predict(X_test)
```

## Cross-validation and Feature Importance
The `fit()` method of the `OPLSCrossValidator` class can be used to evaluate the quality of the OPLS model.
Unless specified as the `cv` parameter, a cross-validator is selected based on the values of the target variable. 
If the target is binary or multiclass, `sklearn.model_selection.StratifiedKFold` is used, otherwise 
`sklearn.model_selection.KFold` is used unless k=-1, then `sklearn.model_selection.LeaveOneOut` is used.

### Determination of orthogonal components
The ideal number of components is determined by creating OPLS models at each number of components from `min_n_components`
until the q-squared value does not increase. The q-squared value is determined by k-fold cross-validation.

This can be performed by calling `OPLSCrossValidator.determine_n_components()`

### Coefficient of determination
The overall quality of the fit is measured using the q-squared metric for the k-fold cross-validated data, where
q-squared is defined as the mean r-squared value of the left-out data. A p-value for q-squared is determined using a 
permutation test (specifically `sklearn.model_selection.permutation_test_score`) where the null hypothesis is that the 
q-squared value was arrived at by chance.

### Permutation significance
The significance of each feature is determined by randomly permuting the value in the feature and observing the change 
in the loading for that feature. Initially, `n_inner_permutations` shufflings are performed per feature, creating 100
loadings per feature. When the loading for the non-permuted data lies outside the middle `1 - inner_alpha` percentile of
the loading's values, an additional 500 permutations are performed to get a more precise p-value. The p-value is defined
as the proportion of loadings which fall within the (-p, p) range where p is the canonical loading, as it is 
in `sklearn.feature_selection.permutation_test_score`.

This can be performed by calling `OPLSCrossValidator.determine_significant_features()`.

#### Example
```pythonstub
from pyopls import OPLSCrossValidator
opls_cv = OPLSCrossValidator()  # k = -1 for leave-one-out
opls_cv.fit(X, y)  # test-train split is automatic
opls_cv.n_components_  # number of components removed
opls_cv.q_squared_  # r-squared value of the regression for all left-out parts of data
opls_cv.q_squared_p_value_  # p-value of q-squared
opls_cv.feature_p_values_  # p-values for significance of features
opls_cv.loadings_  # loadings for X of the one-component PLS model.
opls_cv.estimator_  # the OPLS object 

```

## Notes
`pyopls` can only be used with a single target variable. Only a single component is used in the PLS regression.
## Reference
1. Johan Trygg and Svante Wold. Orthogonal projections to latent structures (O-PLS).
   *J. Chemometrics* 2002; 16: 119-128. DOI: [10.1002/cem.695](https://dx.doi.org/10.1002/cem.695)
2. Eugene Edington and Patrick Onghena. "Calculating P-Values" in *Randomization tests*, 4th edition.
   New York: Chapman & Hall/CRC, 2007, pp. 33-53. DOI: [10.1201/9781420011814](https://doi.org/10.1201/9781420011814).
   
