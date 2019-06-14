import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut, cross_val_score, permutation_test_score
from sklearn.utils.multiclass import type_of_target
import warnings

from .opls import OPLS, OPLSDiscriminator


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

    permutation_q_squared_: array [n_splits*n_permutations]
        The R-squared metric for the left-out data for each permutation

    q_squared_p_value_ : float
        The p-value for the permutation test on Q-squared

    accuracy_ : float, accuracy for discrimination

    permutation_accuracy_: array [n_splits*n_permutations]
        The accuracy of the left-out data for each permutation

    accuracy_p_value_: float
        The p-value for the permutation test on accuracy

    roc_auc_ : float, area under ROC curve for discrimination

    permutation_roc_auc_: array [n_splits*n_permutations]
        The area under the ROC curve of the left-out data for each permutation.

    roc_auc_p_value_: float
        The p-value for the permutation test on the are under the ROC curve.

    n_components_ : float
        The optimal number of orthogonal components to remove

    feature_significance_ : array [n_features], type bool
        Whether permuting the feature results in a significantly different loading for that feature in the model.
        Defined as the loading for the non-permuted data being outside the "middle" of the distribution of loadings
        for the permuted data, where the boundaries are a percentile range defined by outer_alpha.

    feature_p_values_ : array [n_features]
        An estimated p-value for the significance of the feature, defined as the ratio of loading values inside (-p,p)
        where p is the loading for non-permuted data.

    permuted_loadings_ : array [n_inner_permutations, n_features]
        Values for the loadings for the permuted data.

    loadings_ : array [n_features]
        Loadings for the non-permuted data

    estimator_ : OPLS
        The OPLS regressor


    References
    ----------
    Johan Trygg and Svante Wold. Orthogonal projections to latent structures (O-PLS).
    J. Chemometrics 2002; 16: 119-128. DOI: 10.1002/cem.695
    """
    def __init__(self,
                 min_n_components=1,
                 k=10,
                 scale=True,
                 n_permutations=100,
                 n_inner_permutations=100,
                 n_outer_permutations=500,
                 inner_alpha=0.2,
                 outer_alpha=0.01):
        self.min_n_components = min_n_components
        self.k = k
        self.scale = scale
        self.n_permutations = n_permutations
        self.n_inner_permutations = n_inner_permutations
        self.n_outer_permutations = n_outer_permutations
        self.inner_alpha = inner_alpha
        self.outer_alpha = outer_alpha
        self.n_components_ = None
        self.feature_significance_ = None
        self.feature_p_values_ = None
        self.q_squared_ = None
        self.permutation_q_squared_ = None
        self.q_squared_p_value_ = None

        self.accuracy_ = None
        self.permutation_accuracy_ = None
        self.accuracy_p_value_ = None
        self.roc_auc_ = None
        self.permutation_roc_auc_ = None
        self.roc_auc_p_value_ = None

        self.permuted_loadings_ = None
        self.estimator_ = None
        self.loadings_ = None

    def _get_validator(self, Y):
        if self.k == -1:
            return LeaveOneOut()
        else:
            if type_of_target(Y) in ('binary', 'multiclass'):
                return StratifiedKFold(self.k)
            else:
                return KFold(self.k)

    @staticmethod
    def _is_discrimination(Y, target_label=None):
        if type_of_target(Y).startswith('multilabel') and target_label is None:
            warnings.warn('Multilabel response with no target specified. Treating as continuous.')
            return False
        return type_of_target(Y).startswith('binary') or type_of_target(Y).startswith('multilabel')

    def _get_scoring(self, Y, target_label=None):
        return 'accuracy' if self._is_discrimination(Y, target_label) else 'neg_mean_squared_error'

    def _get_estimator(self, Y, n_components, target_label=None):
        return OPLSDiscriminator(n_components, self.scale) if self._is_discrimination(Y, target_label) else OPLS(n_components, self.scale)

    def _validate(self, X, Y, n_components, scoring, cv=None, target_label=None):
        cv = cv or self._get_validator(Y)
        return np.sum(cross_val_score(self._get_estimator(Y, n_components, target_label), X, Y, scoring=scoring, cv=cv))

    def determine_n_components(self, X, Y, cv=None, scoring=None, target_label=None):
        """Determine number of orthogonal components to remove.

        Orthogonal components are removed until removing a component does not improve the performance
        of the k-fold cross-validated OPLS estimator, as measured by the residual sum of squares of the left-out
        data.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of predictors.

        Y : array-like, shape = [n_samples, 1]
            Target vector, where n_samples is the number of samples.
            This implementation only supports a single response (target) variable.

        cv : sklearn.model_selection.BaseCrossValidator
            A cross validator. If None, _get_validator() is used to determine the validator. If target is binary or
            multiclass, sklearn.model_selection.StratifiedKFold is used, otherwise sklearn.model_selection.KFold
            is used unless k=-1, then sklearn.model_selection.LeaveOneOut is used.

        scoring :
            Scoring method to use. Will default to 'accuracy' for OPLS-DA and 'neg_mean_squared_error' for OPLS regression.

        target_label : scalar
            If this is a discrimination problem and the response variable is multiclass, the value for the positive
            value in the dummy vector for PLS-DA.


        Returns
        -------
        n_components: int
            The number of components to remove to maximize q-squared

        """
        cv = cv or self._get_validator(Y)
        scoring = self._get_scoring(Y, target_label)
        n_components = self.min_n_components

        score = self._validate(X, Y, n_components, scoring)
        while n_components < X.shape[1]:
            next_score = self._validate(X, Y, n_components + 1, scoring, cv, target_label)
            if next_score <= score:
                break
            else:
                score = next_score
                n_components += 1
        return n_components

    def determine_significant_features(self,
                                       X,
                                       Y,
                                       n_components,
                                       target_label=None):
        """Determine the significance of each feature

        Orthogonal components are removed until removing a component does not improve the performance
        of the k-fold cross-validated OPLS estimator, as measured by the residual sum of squares of the left-out
        data.

        This is always done with a regular OPLS regressor

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of predictors.

        Y : array-like, shape = [n_samples, 1]
            Target vector, where n_samples is the number of samples.
            This implementation only supports a single response (target) variable.

        n_components : int
            The number of orthogonal components to remove

        target_label :
            If this is a discrimination problem, the value of the response used as the positive value in the dummy response.

        Returns
        -------
        significance: array [n_features], type bool
            The number of components to remove to maximize q-squared

        p_values: array [n_features]
            The p-values for each feature. The null hypothesis is that permuting the feature does not change it's weight
            in the one-component PLS model.

        permuted_loadings: array [n_inner_permutations, n_features]
            The one-component PLS loadings for each permutation

        """

        def _get_permuted_loading(n_permutations, j):
            loadings = np.empty(n_permutations, dtype=float)
            for i in range(0, n_permutations):
                X_permuted = np.copy(X)
                X_permuted[:, j] = np.random.permutation(X[:, j])
                test_loadings = estimator.fit(X_permuted, Y).x_loadings_
                # make sure direction of vector is the same
                err1 = np.sum(np.square(test_loadings[:j] - reference_loadings[:j])) \
                       + np.sum(np.square(test_loadings[j:] - reference_loadings[j:]))
                err2 = np.sum(np.square(test_loadings[:j] + reference_loadings[:j])) \
                       + np.sum(np.square(test_loadings[j:] + reference_loadings[j:]))
                sign = -1 if err2 < err1 else 1
                loadings[i] = sign * test_loadings[column].item()
            return loadings

        inside_ptile = self.inner_alpha / 2
        outside_ptile = self.outer_alpha / 2
        p_values = np.empty(X.shape[1], dtype=float)
        significant = np.empty(X.shape[1], dtype=bool)  # filled with True

        # determine loadings for the features using canonical model
        estimator = self._get_estimator(Y, n_components, target_label)
        reference_loadings = np.ravel(estimator.fit(X, Y).x_loadings_)
        loading_max = np.max((reference_loadings, -1 * reference_loadings), axis=0)
        loading_min = np.min((reference_loadings, -1 * reference_loadings), axis=0)
        permuted_loadings = [_get_permuted_loading(self.n_inner_permutations, i) for i in range(0, X.shape[1])]
        for column in range(0, X.shape[1]):
            thresh_min, thresh_max = np.percentile(permuted_loadings[column], (inside_ptile, 1 - inside_ptile))
            if thresh_min <= reference_loadings[column] <= thresh_max:
                p_values[column] = (np.sum(permuted_loadings[column] >= loading_max[column])
                                    + np.sum(permuted_loadings[column] <= loading_min[column]) + 1) / (self.n_inner_permutations + 1)
                significant[column] = False
            else:
                # perform additional permutations if potentially significant
                permuted_loading = _get_permuted_loading(self.n_outer_permutations, column)
                thresh_min, thresh_max = np.percentile(permuted_loading, (outside_ptile, 1 - outside_ptile))
                p_values[column] = (np.sum(permuted_loading >= loading_max[column])
                                    + np.sum(permuted_loading <= loading_min[column]) + 1) / (self.n_outer_permutations + 1)
                significant[column] = not (thresh_min <= reference_loadings[column] <= thresh_max)
        return significant, p_values, np.hstack(permuted_loadings).reshape((self.n_inner_permutations, -1))

    def fit(self, X, Y, n_components=None, cv=None, target_label=None):
        """Evaluate the quality of the OPLS regressor

        The q-squared value and a p-value for each feature's significance is determined. The final regressor can be
        access as estimator_.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of predictors.

        Y : array-like, shape = [n_samples, 1]
            Target vector, where n_samples is the number of samples.
            This implementation only supports a single response (target) variable.

        n_components : int
            The number of orthogonal components to remove. Will be determined by determine_n_components if None

        cv : sklearn.model_selection.BaseCrossValidator
            A cross-validator to use for the determination of the number of components and the q-squared value.
            If None, a cross validator will be selected based on the value of k and the values of the target variable.
            If target is binary or multiclass, sklearn.model_selection.StratifiedKFold is used, otherwise
            sklearn.model_selection.KFold is used unless k=-1, then sklearn.model_selection.LeaveOneOut is used.

        target_label:
            If this is a discrimination problem, the label to use for 1 in the dummy response

        """
        n_components = n_components or self.determine_n_components(X, Y)
        discrimination = self._is_discrimination(Y, target_label)
        cv = cv or self._get_validator(Y)
        # permutation of label to get p-value for q_squared
        self.q_squared_, self.permutation_q_squared_, self.q_squared_p_value_ = permutation_test_score(
            OPLS(n_components, self.scale), X, Y, cv=cv, n_permutations=self.n_permutations
        )
        # if this is a discrimination problem, get accuracy and the AUC of the ROC
        if discrimination:
            self.accuracy_, self.permutation_accuracy_, self.accuracy_p_value_ = permutation_test_score(
                OPLSDiscriminator(n_components, self.scale, target_label=target_label),
                X, Y, cv=cv, scoring='accuracy', n_permutations=self.n_permutations
            )
            self.roc_auc_, self.permutation_roc_auc_, self.roc_auc_p_value_ = permutation_test_score(
                OPLSDiscriminator(n_components, self.scale, target_label=target_label),
                X, Y, cv=cv, scoring='roc_auc', n_permutations=self.n_permutations
            )
        else:
            self.accuracy_ = self.permutation_accuracy_ = self.accuracy_p_value_ = None
            self.roc_auc_ = self.permutation_roc_auc_ = self.roc_auc_p_value_ = None

        (self.feature_significance_,
         self.feature_p_values_,
         self.permuted_loadings_) = self.determine_significant_features(X, Y, n_components)

        if discrimination:
            self.estimator_ = OPLSDiscriminator(n_components, self.scale)
            self.estimator_.fit(X, Y, target_label)
        else:
            self.estimator_ = OPLS(n_components, self.scale)
            self.estimator_.fit(X, Y)

        self.loadings_ = self.estimator_.x_loadings_
        return self
