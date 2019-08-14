from sys import stderr

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import is_classifier, clone
from sklearn.metrics.scorer import _passthrough_scorer
from sklearn.model_selection import check_cv
from sklearn.utils import indexable, check_random_state, safe_indexing
from sklearn.utils.metaestimators import _safe_split


def non_cv_permutation_test_score(estimator, X, y, groups=None,
                                  n_permutations=100, n_jobs=None, random_state=0,
                                  verbose=0, pre_dispatch='2*n_jobs', scorers=None):
    """Evaluate the significance of several non-cross-validated scores with permutations

    Read more in the :ref:`User Guide <cross_validation>`.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    X : array-like of shape at least 2D
        The data to fit.

    y : array-like
        The target variable to try to predict in the case of
        supervised learning.

    groups : array-like, with shape (n_samples,), optional
        Labels to constrain permutation within groups, i.e. ``y`` values
        are permuted among samples with the same group identifier.
        When not specified, ``y`` values are permuted among all samples.

        When a grouped cross-validator is used, the group labels are
        also passed on to the ``split`` method of the cross-validator. The
        cross-validator uses them for grouping the samples  while splitting
        the dataset into train/test set.

    scorers : string, callable or None, optional, default: None
        a list of scoring functions


    n_permutations : integer, optional
        Number of times to permute ``y``.

    n_jobs : int or None, optional (default=None)
        The number of CPUs to use to do the computation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    random_state : int, RandomState instance or None, optional (default=0)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

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
    score : float
        The true score without permuting targets.

    permutation_scores : array, shape (n_permutations,)
        The scores obtained for each permutations.

    pvalue : float
        The p-value, which approximates the probability that the score would
        be obtained by chance. This is calculated as:

        `(C + 1) / (n_permutations + 1)`

        Where C is the number of permutations whose score >= the true score.

        The best possible p-value is 1/(n_permutations + 1), the worst is 1.0.

    Notes
    -----
    This function implements Test 1 in:

        Ojala and Garriga. Permutation Tests for Studying Classifier
        Performance.  The Journal of Machine Learning Research (2010)
        vol. 11

    """
    X, y, groups = indexable(X, y, groups)

    random_state = check_random_state(random_state)
    if scorers is None or not len(scorers):
        if hasattr(estimator, 'score'):
            scorers = [_passthrough_scorer]
        else:
            raise TypeError(
                "If no scoring is specified, the estimator passed should "
                "have a 'score' method. The estimator %r does not."
                % estimator)

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    score = _non_cv_permutation_test_score(clone(estimator), X, y, groups, scorers)
    permutation_scores = Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)(
        delayed(_non_cv_permutation_test_score)(
            clone(estimator), X, _shuffle(y, groups, random_state),
            groups, scorers)
        for _ in range(n_permutations))
    permutation_scores = np.array(permutation_scores)
    pvalue = (np.sum(permutation_scores >= score, axis=0) + 1.0) / (n_permutations + 1)
    return [(score[i], permutation_scores[:, i], pvalue[i]) for i in range(len(scorers))]


def _non_cv_permutation_test_score(estimator, X, y, groups, scorers):
    """Auxiliary function for permutation_test_score"""
    estimator.fit(X, y)
    return [scorer(estimator, X, y) for scorer in scorers]


def permutation_test_score(estimator, X, y, groups=None, cv='warn',
                           n_permutations=100, n_jobs=None, random_state=0,
                           verbose=0, pre_dispatch='2*n_jobs', scorers=None):
    """Evaluate the significance of several cross-validated scores with permutations

    Read more in the :ref:`User Guide <cross_validation>`.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    X : array-like of shape at least 2D
        The data to fit.

    y : array-like
        The target variable to try to predict in the case of
        supervised learning.

    groups : array-like, with shape (n_samples,), optional
        Labels to constrain permutation within groups, i.e. ``y`` values
        are permuted among samples with the same group identifier.
        When not specified, ``y`` values are permuted among all samples.

        When a grouped cross-validator is used, the group labels are
        also passed on to the ``split`` method of the cross-validator. The
        cross-validator uses them for grouping the samples  while splitting
        the dataset into train/test set.

    scorers : string, callable or None, optional, default: None
        a list of scoring functions

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross validation,
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. versionchanged:: 0.20
            ``cv`` default value if None will change from 3-fold to 5-fold
            in v0.22.

    n_permutations : integer, optional
        Number of times to permute ``y``.

    n_jobs : int or None, optional (default=None)
        The number of CPUs to use to do the computation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    random_state : int, RandomState instance or None, optional (default=0)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

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
    score : float
        The true score without permuting targets.

    permutation_scores : array, shape (n_permutations,)
        The scores obtained for each permutations.

    pvalue : float
        The p-value, which approximates the probability that the score would
        be obtained by chance. This is calculated as:

        `(C + 1) / (n_permutations + 1)`

        Where C is the number of permutations whose score >= the true score.

        The best possible p-value is 1/(n_permutations + 1), the worst is 1.0.

    Notes
    -----
    This function implements Test 1 in:

        Ojala and Garriga. Permutation Tests for Studying Classifier
        Performance.  The Journal of Machine Learning Research (2010)
        vol. 11

    """
    X, y, groups = indexable(X, y, groups)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    random_state = check_random_state(random_state)
    if scorers is None:
        if hasattr(estimator, 'score'):
            scorers = [_passthrough_scorer]
        else:
            raise TypeError(
                "If no scoring is specified, the estimator passed should "
                "have a 'score' method. The estimator %r does not."
                % estimator)

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    score = _permutation_test_score(clone(estimator), X, y, groups, cv, scorers)
    permutation_scores = Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)(
        delayed(_permutation_test_score)(
            clone(estimator), X, _shuffle(y, groups, random_state),
            groups, cv, scorers)
        for _ in range(n_permutations))
    permutation_scores = np.array(permutation_scores)
    pvalue = (np.sum(permutation_scores >= score, axis=0) + 1.0) / (n_permutations + 1)
    return [(score[i], permutation_scores[:, i], pvalue[i]) for i in range(len(scorers))]
    # return score, permutation_scores, pvalue


def _permutation_test_score(estimator, X, y, groups, cv, scorers):
    """Auxiliary function for permutation_test_score"""
    avg_score = []
    for train, test in cv.split(X, y, groups):
        X_train, y_train = _safe_split(estimator, X, y, train)
        X_test, y_test = _safe_split(estimator, X, y, test, train)
        estimator.fit(X_train, y_train)
        avg_score.append([scorer(estimator, X_test, y_test) for scorer in scorers])
    return np.mean(avg_score, axis=0)


def _shuffle(y, groups, random_state):
    """Return a shuffled copy of y eventually shuffle among same groups."""
    if groups is None:
        indices = random_state.permutation(len(y))
    else:
        indices = np.arange(len(groups))
        for group in np.unique(groups):
            this_mask = (groups == group)
            indices[this_mask] = random_state.permutation(indices[this_mask])
    return safe_indexing(y, indices)


def feature_permutation_loading(estimator, X, y, initial_permutations=100, alpha=0.2, final_permutations=500,
                                random_state=0, n_jobs=None, verbose=0, pre_dispatch='2*n_jobs'):
    """Determine the significance of each feature

    This is done by permuting each feature in X and measuring the loading.
    The feature is considered significant if the loadings are signficantly different.

    This is always done with a regular OPLS regressor
    OPLS-DA should be binarized first.

    Parameters
    ----------
    estimator : estimator object implementing 'fit' with x_loadings_
        The object to use to fit the data. This should have an [n_features, 1] x_loadings_ array. This can be a
        one-component PLS or OPLS model.

    X : array-like, shape = [n_samples, n_features]
        Training vectors, where n_samples is the number of samples and
        n_features is the number of predictors.

    y : array-like, shape = [n_samples, 1]
        Target vector, where n_samples is the number of samples.
        This implementation only supports a single response (target) variable.

    initial_permutations : int
        The number of permutations to perform for all features.

    alpha : float, in range (0, 1)
        The threshold for significance. If a feature is found significant in the first round, it will be retested with
        final_permutations in the second round.

    final_permutations : int
        The number of permutations to perform during the second round to retest points found significant in the first
        round.

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

    random_state : int, RandomState instance or None, optional (default=0)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Returns
    -------
    x_loadings : array [n_features]
        The x_loadings found from non-permuted data.

    permutation_x_loadings: array [n_inner_permutations, n_features]
        The one-component PLS loadings for each permutation in the first round.

    p_values: array [n_features]
        The p-values for each feature. The null hypothesis is that permuting the feature does not change it's weight
        in the one-component PLS model.
    """

    def feature_ind_generator(n_permutations_, feature_inds):
        """
        Repeats each value in feature_inds n_permutations_ times.
        """
        i = 0
        count = 0
        while count < (n_permutations_ * len(feature_inds)):
            yield feature_inds[i]
            count += 1
            if (count % n_permutations_) == 0:
                i += 1

    def _log(txt):
        if verbose in range(1, 51):
            stderr.write(txt + '\n')
        if verbose > 50:
            print(txt)

    random_state = check_random_state(random_state)
    n_features = X.shape[1]
    x_loadings = np.ravel(estimator.fit(X, y).x_loadings_)
    loading_max = np.max((x_loadings, -1 * x_loadings), axis=0)
    loading_min = np.min((x_loadings, -1 * x_loadings), axis=0)

    _log('Performing initial permutation tests.')
    permutation_x_loadings = Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)(
        delayed(_feature_permutation_loading)(
            clone(estimator), _feature_shuffle(X, feature_ind, random_state), y, x_loadings, feature_ind)
        for feature_ind in feature_ind_generator(initial_permutations, [i for i in range(n_features)]))
    permutation_x_loadings = np.array(permutation_x_loadings).reshape(n_features, initial_permutations).T

    _log('Calculating p values.')
    p_values = Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)(
        delayed(_loading_p_value)(permutation_x_loading, upper, lower, initial_permutations)
        for permutation_x_loading, upper, lower in zip(np.hsplit(permutation_x_loadings, n_features),
                                                       loading_max, loading_min)
    )

    # Retest values found significant in first round
    retest_columns = [i for i in range(n_features) if p_values[i] < (alpha / 2.0)]  # remember, this is two-tailed
    retest_loading_max = np.max((x_loadings[retest_columns], -1 * x_loadings[retest_columns]), axis=0)
    retest_loading_min = np.min((x_loadings[retest_columns], -1 * x_loadings[retest_columns]), axis=0)

    _log(f'Re-testing {len(retest_columns)} features')
    retest_loadings = Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)(
        delayed(_feature_permutation_loading)(
            clone(estimator), _feature_shuffle(X, feature_ind, random_state), y, x_loadings, feature_ind)
        for feature_ind in feature_ind_generator(final_permutations, retest_columns))
    retest_loadings = np.array(retest_loadings).reshape(len(retest_columns), final_permutations).T

    # replace p-values with the more accurate ones
    _log(f'Calculating p values for {len(retest_columns)} features.')
    p_values = np.array(p_values)
    p_values[retest_columns] = Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)(
        delayed(_loading_p_value)(retest_loading, upper, lower, initial_permutations)
        for retest_loading, upper, lower in zip(np.hsplit(retest_loadings, len(retest_columns)),
                                                retest_loading_max, retest_loading_min)
    )
    p_values = np.array(p_values)
    return x_loadings, permutation_x_loadings, p_values


def _feature_permutation_loading(estimator, X, y, reference_loadings, feature_ind):
    """Auxiliary function for feature_permutation_loading"""
    """Not that since loading only depends on training data, we dont use cross-validation"""
    test_loadings = np.ravel(estimator.fit(X, y).x_loadings_)
    # make directions the same
    err1 = (np.sum(np.square(test_loadings[:feature_ind] - reference_loadings[:feature_ind]))
            + np.sum(np.square(test_loadings[feature_ind:] - reference_loadings[feature_ind:])))
    err2 = (np.sum(np.square(test_loadings[:feature_ind] + reference_loadings[:feature_ind]))
            + np.sum(np.square(test_loadings[feature_ind:] + reference_loadings[feature_ind:])))
    sign = -1 if err2 < err1 else 1
    return sign * test_loadings[feature_ind]


def _feature_shuffle(X, feature_ind, random_state):
    X = X.copy()
    random_state.shuffle(X[:, feature_ind])
    return X


def _loading_p_value(permutation_loadings, upper, lower, n_permutations):
    return (np.sum(permutation_loadings >= upper) + np.sum(permutation_loadings <= lower) + 1) / (n_permutations + 1)

