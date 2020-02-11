import warnings
from sys import stderr

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, ClassifierMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, r2_score
from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut, cross_val_predict
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_array
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted

from .opls import OPLS
from .permutation_test import permutation_test_score, feature_permutation_loading


def discriminator_accuracy(y_true, y_pred):
    try:
        return accuracy_score(y_true.astype(int), np.sign(y_pred).astype(int))
    except ValueError as e:
        warnings.warn(str(e), UserWarning)
        return float('nan')


def discriminator_roc_auc(y_true, y_pred):
    try:
        return roc_auc_score(y_true, np.clip(y_pred, -1, 1))
    except ValueError as e:
        warnings.warn(str(e), UserWarning)
        return float('nan')


def discriminator_r2_score(y_true, y_pred):
    return r2_score(y_true, np.clip(y_pred, -1, 1))


def neg_press(y_true, y_pred):
    return -1 * np.sum(np.square(y_true - y_pred))


def neg_pressd(y_true, y_pred):
    return -1 * np.sum(np.square(y_true - np.clip(y_pred, -1, 1)))


class OPLSValidator(BaseEstimator, TransformerMixin, RegressorMixin):
    """Cross Validation and Diagnostics of Orthogonal Projection to Latent Structures (O-PLS)

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


    r_squared_Y_: float, overall R-squared metric for the regression

    r_squared_X_: float, overall R-squared X metric (

    discriminant_q_squared_: float
        Discriminant Q-squared, if this is an OPLSDA problem. Discriminant Q-squared disregards the error of class
        predictions whose values are beyond the class labels (e.g. it treats predictions of -1.5 as -1 and 1.5 as 1).

    permutation_discriminant_q_squared_: array [n_splits*n_permutations]
        The discriminant R-squared metric for the left-out data for each permutation.

    discriminant_q_squared_p_value_ : float
        The p-value for the permutation test on DQ-squared

    accuracy_ : float, accuracy for discrimination

    discriminant_r_squared_: float
        Discriminant R-squared, if this is an OPLSDA problem. Discriminant R-squared disregards the error of class
        predictions whose values are beyond the class labels (e.g. it treats a predictions of -1.5 as -1 and 1.5 as 1).

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

    opls_ : OPLS
        The OPLS transformer

    pls_ : PLSRegression
        A 1-component PLS regressor used to evaluate the OPLS transform


    References
    ----------
    Johan Trygg and Svante Wold. Orthogonal projections to latent structures (O-PLS).
    J. Chemometrics 2002; 16: 119-128. DOI: 10.1002/cem.695

    Johan A. Westerhuis, Ewoud J. J. van Velzen, Huub C. J. Hoefsloot and Age K. Smilde.
    Discriminant Q-squared (DQ-squared) for improved discrimination in PLSDA models.
    Metabolomics (2008) 4: 293. https://doi.org/10.1007/s11306-008-0126-2
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
                 outer_alpha=0.05):
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

        self.r_squared_Y_ = None
        self.discriminant_r_squared_ = None
        self.r_squared_X_ = None

        self.q_squared_ = None
        self.permutation_q_squared_ = None
        self.q_squared_p_value_ = None

        self.accuracy_ = None
        self.permutation_accuracy_ = None
        self.accuracy_p_value_ = None

        self.roc_auc_ = None
        self.permutation_roc_auc_ = None
        self.roc_auc_p_value_ = None

        self.discriminant_q_squared_ = None
        self.discriminant_q_squared_p_value_ = None
        self.permutation_discriminant_q_squared_ = None

        self.permutation_loadings_ = None
        self.pls_ = None  # a 1-component PLSRegression
        self.opls_ = None  # OPLS transform
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

    def _get_score_function(self, Y):
        return neg_pressd if self.is_discrimination(Y) else neg_press

    def _validate(self, X, Y, n_components, score_function, cv=None, n_jobs=None, verbose=0, pre_dispatch='2*n_jobs'):
        cv = cv or self._get_validator(Y, self.k)
        Z = OPLS(n_components, self.scale).fit_transform(X, Y)
        y_pred = cross_val_predict(PLSRegression(1, self.scale), Z, Y, cv=cv, n_jobs=n_jobs, verbose=verbose,
                                   pre_dispatch=pre_dispatch)
        return score_function(Y, y_pred)

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
        scoring = scoring or self._get_score_function(y)
        n_components = self.min_n_components

        score = self._validate(X, y, n_components, scoring, cv, n_jobs, verbose, pre_dispatch)
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
                                        random_state=0,
                                        n_jobs=None,
                                        verbose=0,
                                        pre_dispatch='2*n_jobs'):
        """Determine the significance of each feature

        This is done by permuting each feature in X and measuring the loading.
        The feature is considered significant if the loadings are significantly different.

        This is always done with a regular PLS regressor
        PLS-DA should be binarized first.

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
            Whether a particular feature is significant.

        p_values: array [n_features]
            The p-values for each feature. The null hypothesis is that permuting the feature does not change it's weight
            in the one-component PLS model.

        permuted_loadings: array [n_inner_permutations, n_features]
            The one-component PLS loadings for each permutation
        """
        Z = OPLS(n_components, self.scale).fit_transform(X, y)
        x_loadings, permutation_x_loadings, p_values = feature_permutation_loading(
            PLSRegression(1, self.scale), Z, y, self.n_inner_permutations, self.inner_alpha,
            self.n_outer_permutations, random_state, n_jobs, verbose, pre_dispatch
        )
        return p_values < self.outer_alpha, p_values, permutation_x_loadings

    def cross_val_roc_curve(self, X, y, cv=None, n_jobs=None, verbose=0, pre_dispatch='2*n_jobs'):
        Z = self.opls_.transform(X)
        cv = cv or self._get_validator(y, self.k)
        check_is_fitted(self, ['opls_', 'pls_', 'binarizer_'])
        y_pred = cross_val_predict(PLSRegression(1, self.scale), Z, y, cv=cv, n_jobs=n_jobs, verbose=verbose,
                                   pre_dispatch=pre_dispatch)
        return roc_curve(y, y_pred)

    def fit(self, X, y, n_components=None, cv=None, pos_label=None,
            random_state=0, n_jobs=None, verbose=0, pre_dispatch='2*n_jobs'):
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

        random_state : int, RandomState instance or None, optional (default=0)
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used
            by `np.random`.

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

        def _log(txt):
            if verbose in range(1, 51):
                stderr.write(txt + '\n')
            if verbose > 50:
                print(txt)

        X = check_array(X, dtype=float, copy=True)
        y = self._check_target(y, pos_label)

        if not n_components:
            _log('Determining number of components to remove.')
            n_components = self._determine_n_components(X, y, cv, n_jobs=n_jobs, verbose=verbose,
                                                        pre_dispatch=pre_dispatch)
            _log(f'Removing {n_components} orthogonal components.')
        self.n_components_ = n_components or self._determine_n_components(X, y)

        self.opls_ = OPLS(self.n_components_, self.scale).fit(X, y)
        Z = self.opls_.transform(X)
        self.pls_ = PLSRegression(1, self.scale).fit(Z, y)
        self.r_squared_X_ = self.opls_.score(X)
        y_pred = self.pls_.predict(Z)
        self.r_squared_Y_ = r2_score(y, y_pred)
        if self.is_discrimination(y):
            self.discriminant_r_squared_ = r2_score(y, np.clip(y_pred, -1, 1))

        cv = cv or self._get_validator(y, self.k)

        score_functions = [r2_score]
        if self.is_discrimination(y):
            score_functions += [discriminator_r2_score, discriminator_accuracy, discriminator_roc_auc]

        _log('Performing cross-validated metric permutation tests.')

        cv_results = permutation_test_score(PLSRegression(1, self.scale), Z, y, cv=cv,
                                            n_permutations=self.n_permutations, cv_score_functions=score_functions,
                                            n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)
        if self.is_discrimination(y):
            [
                (self.q_squared_, self.permutation_q_squared_, self.q_squared_p_value_),
                (self.discriminant_q_squared_, self.permutation_discriminant_q_squared_,
                 self.discriminant_q_squared_p_value_),
                (self.accuracy_, self.permutation_accuracy_, self.accuracy_p_value_),
                (self.roc_auc_, self.permutation_roc_auc_, self.roc_auc_p_value_)
            ] = cv_results
        else:
            [
                (self.q_squared_, self.permutation_q_squared_, self.q_squared_p_value_)
            ] = cv_results

        _log('Estimating feature significance.')

        (self.feature_significance_,
         self.feature_p_values_,
         self.permutation_loadings_) = self._determine_significant_features(X, y, self.n_components_, random_state,
                                                                            n_jobs, verbose, pre_dispatch)
        return self

    def transform(self, X):
        return self.opls_.transform(X)

    def predict(self, X):
        Z = self.transform(X)
        return self.pls_.predict(Z)

    def score(self, X, y, sample_weight=None):
        Z = self.transform(X)
        return r2_score(y, self.pls_.predict(Z))

    def discriminator_roc(self, X, y):
        Z = self.transform(X)
        return roc_curve(y, self.pls_.predict(Z))


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
        Z = self.transform(X)
        y_pred = self.pls_.predict(Z)
        return r2_score(y, np.clip(y_pred, -1, 1))

    def predict(self, X):
        Z = self.opls_.transform(X)
        values = np.sign(self.pls_.predict(Z))
        return self.binarizer_.inverse_transform(values).reshape(-1, 1)
