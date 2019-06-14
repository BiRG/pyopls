# Copyright (c) 2019 Wright State University
# Author: Daniel Foose <foose.3@wright.edu>
# License: MIT

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, FLOAT_DTYPES


class OPLS(BaseEstimator, TransformerMixin, RegressorMixin):
    """Orthogonal Projection to Latent Structures (O-PLS)

    This class implements the O-PLS algorithm for one (and only one) response as described by [Trygg 2002].
    This is based on the MATLAB implementation by Paul E. Anderson (https://github.com/Anderson-Lab/OPLS).

    Parameters
    ----------
    n_components: int, number of orthogonal components to filter. (default 5).

    scale: boolean, scale data? (default True)

    copy: boolean
        copy X and Y to new matrices? (default True). Note that your inputs arrays will change if this is False.

    Attributes
    ----------
    orthogonal_x_weights_ : array, [n_features, n_components]
        X block weights vector

    x_weights_ : array, [n_features, 1]
        X block weights vector for first component of filtered data

    y_weights_ : float
        Y block weight (is a scalar because singular Y is required)

    orthogonal_x_loadings_ : array, [n_features, n_components]
        X block loadings vectors

    x_loadings_ : array, [n_features, 1]
        X block loadings vector for first component of filtered data

    orthogonal_x_scores_ : array, [n_samples, n_components]
        X scores for orthogonal part of data.

    x_scores_ : array, [n_samples, 1]
        X scores for the first component of filtered data

    y_scores_ : array, [n_samples, 1]
        Y scores

    coef_ : array, [n_features, 1]
        The coefficients of the linear model created from the filtered data

    R_squared_X_ : float
        R^2 value for X. The amount of X variation in the X data explained by the model. This variation is independent
        of the classes and likely to be noise. This should be smaller in an O-PLS model than in a typical PLS model.

    R_squared_Y_ : float
        R^2 value for Y. The amount of Y variation in the X data explained by the model. This is the value most commonly
        called R-squared. To get the R^2Y value for another X-Y pair, use score().
    
    References
    ----------
    Johan Trygg and Svante Wold. Orthogonal projections to latent structures (O-PLS).
    J. Chemometrics 2002; 16: 119-128. DOI: 10.1002/cem.695
    """
    def __init__(self, n_components=5, scale=True, copy=True):
        self.n_components = n_components
        self.scale = scale
        self.copy = copy
        self.b_ = None
        self.y_weights_ = None  # c
        self.y_scores_ = None  # u
        self.x_scores_ = None  # t
        self.x_loadings_ = None  # p
        self.x_weights_ = None  # w
        self.orthogonal_x_scores_ = None  # t_ortho
        self.orthogonal_x_loadings_ = None  # p_ortho
        self.orthogonal_x_weights_ = None  # w_ortho
        self.coef_ = None  # B_pls
        self.x_mean_ = None
        self.y_mean_ = None
        self.x_std_ = None
        self.y_std_ = None
        self.sum_sq_X_ = None
        self.sum_sq_Y_ = None
        self.R_squared_X_ = None
        self.R_squared_Y_ = None

    @staticmethod
    def _center_scale_xy(X, Y, scale=True):
        """Center X, Y and scale if the scale parameter==True

        Returns
        -------
            X, Y, x_mean, y_mean, x_std, y_std
        """
        x_mean = X.mean(axis=0)
        X -= x_mean
        y_mean = Y.mean(axis=0)
        Y -= y_mean
        if scale:
            x_std = X.std(axis=0, ddof=1)
            x_std[x_std == 0.0] = 1.0
            X /= x_std
            y_std = Y.std(axis=0, ddof=1)
            y_std[y_std == 0] = 1.0
            Y /= y_std
        else:
            x_std = np.ones(X.shape[1])
            y_std = np.ones(Y.shape[1])
        return X, Y, x_mean, y_mean, x_std, y_std

    def fit(self, X, Y):
        """Fit model to data

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of predictors.

        Y : array-like, shape = [n_samples, 1]
            Target vector, where n_samples is the number of samples.
            This implementation only supports a single response (target) variable.

        """
        w_ortho = np.zeros((np.asarray(X).shape[1], self.n_components))
        p_ortho = np.zeros((np.asarray(X).shape[1], self.n_components))
        t_ortho = np.zeros((len(X), self.n_components))
        X = check_array(X, copy=self.copy)
        Y = check_array(Y, copy=self.copy)

        if Y.shape != (X.shape[0], 1):
            raise ValueError('This OPLS implementation does not support multiple Y. '
                             'Y must be a (n_samples, 1) array-like.')

        X_res, Y_res, self.x_mean_, self.y_mean_, self.x_std_, self.y_std_ = self._center_scale_xy(X, Y, self.scale)

        X_res = np.array(X - np.mean(X, 0))  # mean-center X
        Y_res = np.array(Y - np.mean(Y, 0))  # mean-center Y
        SS_Y = np.sum(np.square(Y_res))
        SS_X = np.sum(np.square(X_res))

        for i in range(0, self.n_components):
            # find PLS component
            w = ((Y_res.T @ X_res) / (Y_res.T @ Y_res)).T
            w = w / np.linalg.norm(w)
            t = (X_res @ w) / (w.T @ w)
            p = ((t.T @ X_res) / (t.T @ t)).T

            # run OSC filter on Xres
            w_ortho[:, i] = np.ravel(p - float((w.T @ p) / (w.T @ w)) * w)
            w_ortho[:, i] = w_ortho[:, i] / np.linalg.norm(w_ortho[:, i])
            t_ortho[:, i] = (X_res @ w_ortho[:, i]) / (w_ortho[:, i].T @ w_ortho[:, i])
            p_ortho[:, i] = np.ravel((t_ortho[:, i].T @ X_res) / (t_ortho[:, i].T @ t_ortho[:, i]))
            X_res = X_res - t_ortho[:, i][:, np.newaxis] @ p_ortho[:, i][np.newaxis, :]

        # PLS on full data
        # find first PLS component
        w = ((Y_res.T @ X_res) / (Y_res.T @ Y_res)).T
        w = w / np.linalg.norm(w)
        t = (X_res @ w) / (w.T @ w)
        c = (((t.T @ Y_res) / (t.T @ t)).T).item()  # this only works with single-column y
        u = (Y_res * c) / (c ** 2)
        p = ((t.T @ X_res) / (t.T @ t)).T
        # b coef, whatever that is (it's part of the calculation of the coefs, possibly a y-loading?)
        b_l = ((1.0 / (t.T @ t)) * (u.T @ t)).item()

        self.b_ = b_l  # not sure what "b" is really...
        # self.y_loadings_= ??
        self.y_weights_ = c
        self.y_scores_ = u
        self.x_scores_ = t
        self.x_loadings_ = p
        self.x_weights_ = w
        self.orthogonal_x_scores_ = t_ortho
        self.orthogonal_x_loadings_ = p_ortho
        self.orthogonal_x_weights_ = w_ortho
        self.sum_sq_X_ = SS_X
        self.sum_sq_Y_ = SS_Y

        # Original space
        W_star = w * (1.0 / (p.T @ w))
        B_pls = (W_star * b_l * c)
        self.coef_ = B_pls.reshape((B_pls.size, 1))

        self.R_squared_X_ = ((t.T @ t) * (p.T @ p) / SS_X).item()
        self.R_squared_Y_ = ((t.T @ t) * (b_l ** 2.0) * (c ** 2.0) / SS_Y).item()
        return self

    def get_nonorthogonal(self, X):
        """Get the non-orthogonal components of X (which are considered in prediction).

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training or test vectors, where n_samples is the number of samples and
            n_features is the number of predictors (which should be the same predictors the model was trained on).

        Returns
        -------
        X_res, X with the orthogonal data filtered out
        """
        X = check_array(X)
        z = X - X.mean(axis=0)

        # filter out orthogonal components of X
        for f in range(0, self.n_components):
            z = (z - (z @ self.orthogonal_x_weights_[:, f][:, np.newaxis] /
                      (self.orthogonal_x_weights_[:, f].T @ self.orthogonal_x_weights_[:, f]))
                 @ self.orthogonal_x_loadings_[:, f][np.newaxis, :])
        return z

    def predict(self, X):
        """Apply the dimension reduction learned on the training data.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training or test vectors, where n_samples is the number of samples and
            n_features is the number of predictors (which should be the same predictors the model was trained on).

        Notes
        -----
        Unlike in sklearn.cross_decomposition.PLSRegression, the prediction from X cannot modify X
        """
        check_is_fitted(self, 'x_mean_')
        z = self.get_nonorthogonal(X)
        return np.dot(z, self.coef_) + self.y_mean_

    def transform(self, X, Y=None, copy=False):
        """Get the PLS scores for the data that was removed

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of predictors.

        Y : array-like, shape = [n_samples, 1]
            Target vector, where n_samples is the number of samples.
            This implementation only supports a single response (target) vector.

        Returns
        -------
        x_scores if Y is not given, (x_scores, y_scores) otherwise.

        Notes
        -----
        This is not a PLS transformation of the original data. To get that information, run a
        sklearn.cross_decomposition.PLSRegression on the results of get_nonorthogonal.
        """
        check_is_fitted(self, 'x_mean_')
        X = check_array(X, dtype=FLOAT_DTYPES, copy=copy)
        x_scores = np.dot((X - self.x_mean_) / self.x_std_, self.orthogonal_x_weights_)
        if Y is not None:
            Y = check_array(Y)  # will throw for 1d Y
            y_scores = np.dot((Y - self.y_mean_) / self.y_std_, self.y_weights_)
            return x_scores, y_scores
        return x_scores

    def fit_transform(self, X, y=None, **fit_params):
        """ Learn and apply the dimension reduction on the training data and get the scores for the data that is removed

        Parameters
        ----------
        X : array-like, shape=[n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of predictors.

        y : array-like, shape = [n_samples, 1]
            Target vector, where n_samples is the number of samples.
            This O-PLS implementation only supports a single response (target) variable.

        Returns
        -------
        x_scores if Y is not given, (x_scores, y_scores) otherwise.
        """
        return self.fit(X, y).transform(X, y)

