import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut, cross_val_score
from sklearn.model_selection import permutation_test_score
from sklearn.utils.multiclass import type_of_target
from eli5.sklearn import PermutationImportance

from .opls import OPLS


class OPLSCrossValidator:
    """Cross Validation of Orthogonal Projection to Latent Structures (O-PLS)

    This class implements the O-PLS algorithm for one (and only one) response as described by [Trygg 2002].
    This is based on the MATLAB implementation by Paul E. Anderson (https://github.com/Anderson-Lab/OPLS).

    Parameters
    ----------
    min_n_components : int, minimum number of orthogonal components to remove

    k : int
        number of folds for k-fold cross-validation (default 5). If set to -1, leave-one out cross-validation is used.

    scale : boolean, scale data? (default True)

    n_permutations : int, number of permutations to perform on X

    Attributes
    ----------
    q_squared_: float, overall Q-squared metric for the regression, the R-squared value of the left-out data.

    permutation_q_squared_: array [n_splits]
        The R-squared metric for the left-out data for each permutation

    p_value_ : float
        The p-value which approximates whether the Q-squared value was obtained by chance.


    References
    ----------
    Johan Trygg and Svante Wold. Orthogonal projections to latent structures (O-PLS).
    J. Chemometrics 2002; 16: 119-128. DOI: 10.1002/cem.695
    """
    def __init__(self, min_n_components=1, k=10, scale=True, n_permutations=100):
        self.min_n_components = min_n_components
        self.k = k
        self.scale = scale
        self.n_permutations = n_permutations
        self.q_squared_ = None
        self.permutation_q_squared_ = None
        self.p_value_ = None

    def _get_validator(self, Y):
        if self.k == -1:
            return LeaveOneOut()
        else:
            if type_of_target(Y) in ('binary', 'multiclass'):
                return StratifiedKFold(self.k)
            else:
                return KFold(self.k)

    @staticmethod
    def _press(estimator, X, y):
        y_pred = estimator.predict(X)
        return np.sum(np.square(y - y_pred))

    def _validate(self, X, Y, n_components, scoring):
        opls = OPLS(n_components, self.scale)
        cv = self._get_validator(Y)
        return np.sum(cross_val_score(opls, X, Y, scoring=scoring, cv=cv))

    def determine_n_components(self, X, Y):
        """Determine number of orthogonal components to remove.

        Orthogonal components are removed until removing a component does not improve the performance
        of the k-fold cross-validated OPLS estimator, as measured by the residual sum of squares of the left-out
        data.


        """
        n_components = self.min_n_components
        press = self._validate(X, Y, n_components, OPLSCrossValidator._press)
        while n_components < X.shape[1]:
            next_press = self._validate(X, Y, n_components + 1, OPLSCrossValidator._press)
            if next_press/press >= 1:
                break
            else:
                press = next_press
                n_components += 1
        return n_components

    def fit(self, X, Y):
        # permutation of label to get p-value for accuracy
        n_components = self.determine_n_components(X, Y)
        opls = OPLS(n_components, self.scale)
        cv = self._get_validator(Y)
        q_squared, permutation_q_squared, q_squared_p_value = permutation_test_score(
            opls, X, Y, scoring='r2', cv=cv, n_permutations=self.n_permutations
        )
        # determine significant features
        perm = PermutationImportance(
            opls,  n_iter=self.n_permutations, cv=self._get_validator(Y), scoring='accuracy'
        )

        perm.fit(X, Y)
        self.feature_importances_ = perm.feature_importances_
        self.feature_importances_std_ = perm.feature_importances_std_
        perm_worse = np.sum(np.vstack(perm.results_) < q_squared, axis=1) / len(perm.results_)



