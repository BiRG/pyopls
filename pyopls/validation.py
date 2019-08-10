import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, ClassifierMixin
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut, cross_val_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_array
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted

from .opls import OPLS
from .permutation_test import permutation_test_score


class OPLSValidator(BaseEstimator, TransformerMixin, RegressorMixin):
    """Cross Validation and Diagnostics of Orthogonal Projection to Latent Structures (O-PLS)

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

    permutation_loadings_ : array [n_inner_permutations, n_features]
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
                 force_regression=False,
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
        self.force_regression = force_regression
        self.n_components_ = None
        self.feature_significance_ = None
        self.feature_p_values_ = None

        self.q_squared_ = None
        self.permutation_q_squared_ = None
        self.q_squared_p_value_ = None
        self.r_squared_Y_ = None
        self.permutation_r_squared_Y_ = None
        self.r_squared_Y_p_value_ = None

        self.r_squared_X_ = None
        self.permutation_r_squared_X_ = None
        self.r_squared_X_p_value_ = None

        self.accuracy_ = None
        self.permutation_accuracy_ = None
        self.accuracy_p_value_ = None
        self.roc_auc_ = None
        self.permutation_roc_auc_ = None
        self.roc_auc_p_value_ = None

        self.discriminator_q_squared_ = None
        self.discriminator_q_squared_p_value_ = None
        self.permutation_discriminator_q_squared_ = None

        self.discriminator_r_squared_ = None

        self.permutation_loadings_ = None
        self.estimator_ = None
        self.loadings_ = None
        self.binarizer_ = None

    @staticmethod
    def _get_validator(Y, k):
        if k == -1:
            return LeaveOneOut()
        else:
            if type_of_target(Y) in ('binary', 'multiclass'):
                return StratifiedKFold(k)
            else:
                return KFold(k)

    def is_discrimination(self, Y):
        return type_of_target(Y).startswith('binary') and not self.force_regression

    def _get_scoring(self, Y):
        return self._neg_pressd if self.is_discrimination(Y) else self._neg_press

    def _validate(self, X, Y, n_components, scoring, cv=None, n_jobs=None, verbose=0, pre_dispatch='2*n_jobs'):
        cv = cv or self._get_validator(Y, self.k)
        return np.sum(cross_val_score(OPLS(n_components, self.scale),
                                      X, Y, scoring=scoring, cv=cv,
                                      n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch))

    @staticmethod
    def _r2_Y(estimator: OPLS, X, y):
        return estimator.r_squared_Y_

    @staticmethod
    def _r2_X(estimator: OPLS, X, y):
        return estimator.r_squared_X_

    @staticmethod
    def _q2_Y(estimator: OPLS, X, y):
        return estimator.score(X, y)

    @staticmethod
    def _q2d_Y(estimator: OPLS, X, y):
        return estimator.r2d_score(X, y)

    @staticmethod
    def _neg_press(estimator: OPLS, X, y):
        return -1 * estimator.press(X, y)

    @staticmethod
    def _neg_pressd(est: OPLS, X, y):
        return -1 * est.pressd(X, y)

    @staticmethod
    def _discriminator_accuracy(est: OPLS, X, y):
        y_pred = np.sign(est.predict(X))
        return accuracy_score(y.astype(int), y_pred.astype(int))

    @staticmethod
    def _discriminator_roc_auc(est: OPLS, X, y):
        y_score = 0.5 * (np.clip(est.predict(X), -1, 1) + 1)
        return roc_auc_score(y, y_score)

    def _process_binary_target(self, y, pos_label=None):
        self.binarizer_ = LabelBinarizer(-1, 1)
        self.binarizer_.fit(y)
        if pos_label is not None and self.binarizer_.transform([pos_label])[0] == -1:
            self.binarizer_.classes_ = np.flip(self.binarizer_.classes_)
        return self.binarizer_.transform(y).astype(float)

    def _check_target(self, y, pos_label=None):
        y = check_array(y, dtype=None, copy=True, ensure_2d=False).reshape(-1, 1)
        if type_of_target(y).startswith('multiclass') and not self.force_regression:
            raise ValueError('Multiclass input not directly supported. '
                             'Try binarizing with sklearn.preprocessing.LabelBinarizer.')
        if self.is_discrimination(y):
            y = self._process_binary_target(y, pos_label)
        else:
            self.binarizer_ = None
        return y

    def _determine_n_components(self, X, y, cv=None, scoring=None, n_jobs=None, verbose=0, pre_dispatch='2*n_jobs'):
        """Determine number of orthogonal components to remove.

        Orthogonal components are removed until removing a component does not improve the performance
        of the k-fold cross-validated OPLS estimator, as measured by the residual sum of squares of the left-out
        data.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of predictors.

        y : array-like, shape = [n_samples, 1]
            Target vector, where n_samples is the number of samples.
            This implementation only supports a single response (target) variable.

        cv : sklearn.model_selection.BaseCrossValidator
            A cross validator. If None, _get_validator() is used to determine the validator. If target is binary or
            multiclass, sklearn.model_selection.StratifiedKFold is used, otherwise sklearn.model_selection.KFold
            is used unless k=-1, then sklearn.model_selection.LeaveOneOut is used.

        scoring :
            Scoring method to use. Will default to 'accuracy' for OPLS-DA and 'neg_mean_squared_error' for OPLS regression.

        n_jobs : int or None, optional (default=None)
            The number of CPUs to use to do the computation.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
            for more details.

        verbose : integer, optional
            The verbosity level.

        pre_dispatch : int, or string, optional
            Controls the number of jobs that get dispatched during parallel
            execution. Reducing this number can be useful to avoid an
            explosion of memory consumption when more jobs get dispatched
            than CPUs can process. This parameter can be:

                - None, in which case all the jobs are immediately
                  created and spawned. Use this for lightweight and
                  fast-running jobs, to avoid delays due to on-demand
                  spawning of the jobs

                - An int, giving the exact number of total jobs that are
                  spawned

                - A string, giving an expression as a function of n_jobs,
                  as in '2*n_jobs'
        Returns
        -------
        n_components: int
            The number of components to remove to maximize q-squared

        """
        cv = cv or self._get_validator(y, self.k)
        scoring = scoring or self._get_scoring(y)
        n_components = self.min_n_components

        score = self._validate(X, y, n_components, scoring)
        while n_components < X.shape[1]:
            next_score = self._validate(X, y, n_components + 1, scoring, cv, n_jobs, verbose, pre_dispatch)
            if next_score <= score:
                break
            else:
                score = next_score
                n_components += 1
        return n_components

    def _determine_significant_features(self,
                                        X,
                                        y,
                                        n_components,
                                        n_jobs=None,
                                        verbose=0,
                                        pre_dispatch='2*n_jobs'):
        """Determine the significance of each feature

        This is done by permuting each feature in X and measuring the loading.
        The feature is considered significant if the loadings are signficantly different.

        This is always done with a regular OPLS regressor
        OPLS-DA should be binarized first.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of predictors.

        y : array-like, shape = [n_samples, 1]
            Target vector, where n_samples is the number of samples.
            This implementation only supports a single response (target) variable.

        n_components : int
            The number of orthogonal components to remove

        n_jobs : int or None, optional (default=None)
            The number of CPUs to use to do the computation.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
            for more details.

        verbose : integer, optional
            The verbosity level.


        pre_dispatch : int, or string, optional
            Controls the number of jobs that get dispatched during parallel
            execution. Reducing this number can be useful to avoid an
            explosion of memory consumption when more jobs get dispatched
            than CPUs can process. This parameter can be:

                - None, in which case all the jobs are immediately
                  created and spawned. Use this for lightweight and
                  fast-running jobs, to avoid delays due to on-demand
                  spawning of the jobs

                - An int, giving the exact number of total jobs that are
                  spawned

                - A string, giving an expression as a function of n_jobs,
                  as in '2*n_jobs'

        Returns
        -------
        significance: array [n_features], type bool
            The number of components to remove to maximize q-squared

        p_values: array [n_features]
            The p-values for each feature. The null hypothesis is that permuting the feature does not change it's weight
            in the one-component PLS model.

        permuted_loadings: array [n_regressors, n_inner_permutations, n_features]
            The one-component PLS loadings for each permutation
        """

        # permuted loadings for one feature
        def _get_permuted_loading(n_permutations, j):
            loadings = np.empty(n_permutations, dtype=float)
            for i in range(0, n_permutations):
                X_permuted = np.copy(X)
                X_permuted[:, j] = np.random.permutation(X[:, j])
                test_loadings = estimator.fit(X_permuted, y).x_loadings_
                # make sure direction of vector is the same
                err1 = np.sum(np.square(test_loadings[:j] - reference_loadings[:j])) \
                       + np.sum(np.square(test_loadings[j:] - reference_loadings[j:]))
                err2 = np.sum(np.square(test_loadings[:j] + reference_loadings[:j])) \
                       + np.sum(np.square(test_loadings[j:] + reference_loadings[j:]))
                sign = -1 if err2 < err1 else 1
                loadings[i] = sign * test_loadings[j].item()
            return loadings

        def _determine_feature_significance(column):
            p_value = ((np.sum(permuted_loadings[column] >= loading_max[column])
                        + np.sum(permuted_loadings[column] <= loading_min[column]) + 1)
                       / (self.n_inner_permutations + 1))
            if p_value < inner_alpha:
                # perform additional permutations if potentially significant
                permuted_loading = _get_permuted_loading(self.n_outer_permutations, column)
                p_value = (np.sum(permuted_loading >= loading_max[column])
                           + np.sum(permuted_loading <= loading_min[column]) + 1) / (self.n_outer_permutations + 1)
            return p_value

        # determine loadings for the features using canonical model
        estimator = OPLS(n_components, self.scale)

        inner_alpha = self.inner_alpha / 2
        outside_alpha = self.outer_alpha / 2

        reference_loadings = np.ravel(estimator.fit(X, y).x_loadings_)
        loading_max = np.max((reference_loadings, -1 * reference_loadings), axis=0)
        loading_min = np.min((reference_loadings, -1 * reference_loadings), axis=0)

        permuted_loadings = Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)(
            delayed(_get_permuted_loading)(self.n_inner_permutations, i) for i in range(X.shape[1]))

        p_values = Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)(
            delayed(_determine_feature_significance)(i) for i in range(X.shape[1]))

        significant = [p_value < outside_alpha for p_value in p_values]

        return np.hstack(significant), np.hstack(p_values), np.vstack(permuted_loadings).T

    def cross_val_roc_curve(self, X, y, cv=None, n_jobs=None, verbose=0, pre_dispatch='2*n_jobs'):
        def _get_roc(train_inds, test_inds):
            probs = self.estimator_.fit(X[train_inds], target[train_inds]).predict_proba(X[test_inds])
            return roc_curve(target[test_inds], probs)

        X = check_array(X, dtype=float, copy=True)
        y = check_array(y, dtype=None, copy=True, ensure_2d=False).reshape(-1, 1)
        cv = cv or self._get_validator(y, self.k)
        check_is_fitted(self, ['estimator_', 'binarizer_'])
        target = self.binarizer_.transform(y).astype(float)
        results = Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)(
            delayed(_get_roc)(train, test) for train, test in cv.split(X, y)
        )
        return [result[0] for result in results], [result[1] for result in results], [result[2] for result in results]

    def fit(self, X, y, n_components=None, cv=None, pos_label=None, n_jobs=None, verbose=0, pre_dispatch='2*n_jobs'):
        """Evaluate the quality of the OPLS regressor

        The q-squared value and a p-value for each feature's significance is determined. The final regressor can be
        accessed as estimator_.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of predictors.

        y : array-like, shape = [n_samples, 1]
            Target vector, where n_samples is the number of samples.
            This implementation only supports a single response (target) variable.

        n_components : int
            The number of orthogonal components to remove. Will be determined by determine_n_components if None

        cv : sklearn.model_selection.BaseCrossValidator
            A cross-validator to use for the determination of the number of components and the q-squared value.
            If None, a cross validator will be selected based on the value of k and the values of the target variable.
            If target is binary or multiclass, sklearn.model_selection.StratifiedKFold is used, otherwise
            sklearn.model_selection.KFold is used unless k=-1, then sklearn.model_selection.LeaveOneOut is used.

        pos_label : string
            If this is a discrimination problem, the value of the target corresponding to "1".

        n_jobs : int or None, optional (default=None)
            The number of CPUs to use to do the computation.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
            for more details.

        verbose : integer, optional
            The verbosity level.


        pre_dispatch : int, or string, optional
            Controls the number of jobs that get dispatched during parallel
            execution. Reducing this number can be useful to avoid an
            explosion of memory consumption when more jobs get dispatched
            than CPUs can process. This parameter can be:

                - None, in which case all the jobs are immediately
                  created and spawned. Use this for lightweight and
                  fast-running jobs, to avoid delays due to on-demand
                  spawning of the jobs

                - An int, giving the exact number of total jobs that are
                  spawned

                - A string, giving an expression as a function of n_jobs,
                  as in '2*n_jobs'

        """

        X = check_array(X, dtype=float, copy=True)
        y = self._check_target(y, pos_label)

        if not n_components:
            if verbose:
                print('Determining number of components to remove.')
            n_components = self._determine_n_components(X, y)

        self.n_components_ = n_components or self._determine_n_components(X, y)
        cv = cv or self._get_validator(y, self.k)

        scorers = [self._r2_Y, self._r2_X, self._q2_Y]
        if self.is_discrimination(y):
            scorers += [self._q2d_Y, self._discriminator_accuracy, self._discriminator_roc_auc]

        if verbose:
            print('Validating metrics.')

        results = permutation_test_score(OPLS(self.n_components_, self.scale), X, y, cv=cv,
                                         n_permutations=self.n_permutations, scorers=scorers, n_jobs=n_jobs,
                                         verbose=verbose)
        if self.is_discrimination(y):
            (
                (self.r_squared_Y_, self.permutation_r_squared_Y_, self.r_squared_Y_p_value_),
                (self.r_squared_X_, self.permutation_r_squared_X_, self.r_squared_X_p_value_),
                (self.q_squared_, self.permutation_q_squared_, self.q_squared_p_value_),
                (self.discriminator_q_squared_, self.permutation_discriminator_q_squared_,
                 self.discriminator_q_squared_p_value_),
                (self.accuracy_, self.permutation_accuracy_, self.accuracy_p_value_),
                (self.roc_auc_, self.permutation_roc_auc_, self.roc_auc_p_value_)
            ) = results
        else:
            (
                (self.r_squared_Y_, self.permutation_r_squared_Y_, self.r_squared_Y_p_value_),
                (self.r_squared_X_, self.permutation_r_squared_X_, self.r_squared_X_p_value_),
                (self.q_squared_, self.permutation_q_squared_, self.q_squared_p_value_),
            ) = results

        if verbose:
            print('Estimating feature significance.')
        (self.feature_significance_,
         self.feature_p_values_,
         self.permutation_loadings_) = self._determine_significant_features(X, y, self.n_components_,
                                                                            n_jobs, verbose, pre_dispatch)

        self.estimator_ = OPLS(self.n_components_, self.scale).fit(X, y)
        if self.is_discrimination(y):
            self.discriminator_r_squared_ = self.estimator_.r2d_score(X, y)
        return self

    def transform(self, X):
        return self.estimator_.transform(X)

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def score(self, X, y, sample_weight=None):
        return self.estimator_.q2_score(X, y)

    def discriminator_roc(self, X, y):
        return roc_curve(y, self.estimator_.predict_proba(X))


class OPLSDAValidator(OPLSValidator, ClassifierMixin):
    def __init__(self,
                 min_n_components=1,
                 k=10,
                 scale=True,
                 force_regression=False,
                 n_permutations=100,
                 n_inner_permutations=100,
                 n_outer_permutations=500,
                 inner_alpha=0.2,
                 outer_alpha=0.01):
        super().__init__(min_n_components,
                         k,
                         scale,
                         force_regression,
                         n_permutations,
                         n_inner_permutations,
                         n_outer_permutations,
                         inner_alpha,
                         outer_alpha)

    def score(self, X, y, sample_weight=None):
        return self.estimator_.discriminator_accuracy_score(X, y)

    def predict(self, X):
        values = np.sign(self.estimator_.predict(X))
        return self.binarizer_.inverse_transform(values).reshape(-1, 1)
