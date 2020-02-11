# pyopls
##### Orthogonal Projection to Latent Structures in Python. 

This package provides a scikit-learn-style regressor to perform OPLS(-DA). OPLS is a pre-processing method to remove
variation from the descriptor variables that are orthogonal to the target variable (1).

This package also provides a class to validate OPLS models using a 1-component PLS regression with cross-validation 
and permutation tests for both regression and classification metrics (from permutations of the target) and feature PLS
loadings (from permutations of the features). 



A 1-component PLS regression is performed to evaluate the filtering.

## Notes:
* The implementation provided here is equivalent to that of the [libPLS MATLAB library](http://libpls.net/), which is a
  faithful recreation of Trygg and Wold's algorithm.
    * This package uses a different definition for R2X, however (see below)
* `OPLS` inherits `sklearn.base.TransformerMixin` (like `sklearn.decomposition.PCA`) but does not inherit 
`sklearn.base.RegressorMixin` because it is not a regressor like `sklearn.cross_decomposition.PLSRegression`. You can 
use the output of `OPLS.transform()` as an input to another regressor or classifier.
* Like `sklearn.cross_decomposition.PLSRegression`, `OPLS` will center both X and Y before performing the algorithm. 
This makes centering by class in PLS-DA models unnecessary.
* The `score()` function of `OPLS` performs the R-squared X score, the ratio of the variance in the transformed X to the
variance in the original X. A lower score indicates more orthogonal variance removed.

## Examples
### Perform OPLS and PLS-DA on wine dataset
```pythonstub
import pandas as pd
import numpy as np
from pyopls import OPLS
from sklearn.datasets import load_wine
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict, LeaveOneOut()
from sklearn.metrics import r2_score, accuracy_score

wine_data = load_wine()
df = pd.DataFrame(wine_data['data'], columns=wine_data['feature_names'])
df['classification'] = wine_data['target']
df = df[df.classification.isin((0, 1))]
target = df.classification.apply(lambda x: 1 if x else -1)  # discriminant for class 1 vs class 0
X = df[[c for c in df.columns if c!='classification']]
opls = OPLS(2)
Z = opls.fit_transform(X, target)

pls = PLSRegression(1)


``` 
### Validation
The `fit()` method of `OPLSValidator` will find the optimum number of components to remove, then evaluate the results on
 a 1-component `sklearn.cross_decomposition.PLSRegression` model.


## Notes
`pyopls` can only be used with a single target variable.
## Reference
1. Johan Trygg and Svante Wold. Orthogonal projections to latent structures (O-PLS).
   *J. Chemometrics* 2002; 16: 119-128. DOI: [10.1002/cem.695](https://dx.doi.org/10.1002/cem.695)
2. Eugene Edington and Patrick Onghena. "Calculating P-Values" in *Randomization tests*, 4th edition.
   New York: Chapman & Hall/CRC, 2007, pp. 33-53. DOI: [10.1201/9781420011814](https://doi.org/10.1201/9781420011814).
3. Johan A. Westerhuis, Ewoud J. J. van Velzen, Huub C. J. Hoefsloot, Age K. Smilde. Discriminant Q-squared for 
   improved discrimination in PLSDA models. *Metabolomics* 2008; 4: 293-296. 
   DOI: [10.1007/s11306-008-0126-2](https://doi.org/10.1007/s11306-008-0126-2)
