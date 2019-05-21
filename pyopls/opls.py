# Copyright (c) 2019 Wright State University
# Author: Daniel Foose <foose.3@wright.edu>
# License: MIT

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, MultiOutputMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, FLOAT_DTYPES


class OPLS(BaseEstimator, TransformerMixin, RegressorMixin, MultiOutputMixin):
    """
    Orthogonal Projection to Latent Structures (OPLS)

    This class implements the OPLS algorithm for one (and only one) response as described by [Trygg 2002]

    Parameters
    ----------
    n_components: int, number of components to keep. (default 2).

    scale: boolean, scale data? (default True)

    Attributes
    ----------
    x_weights_ : array, [n_features, n_components]
        X block weights vector

    y_weights_ : float
        Y block weight (is a scalar because singular Y is required)

    x_loadings_ : array, [n_features, n_components]
        X block loadings vectors

    x_scores_ : array, [n_samples, n_components]
        X scores

    y_scores_ : array, [n_samples, 1]
        Y scores

    coef_ : array, [n_features, 1]
        The coefficients of the linear model

    References
    ----------
    Johan Trygg and Svante Wold. Orthogonal projections to latent structures (O-PLS).
    J. Chemometrics 2002; 16: 119-128. DOI: 10.1002/cem.695
    """
    def __init__(self, n_components=2, scale=True):
        """
        :param n_components: number of components to keep
        :param scale: scale data?
        """
        self.b_ = None
        self.y_weights_ = None  # c
        self.x_loadings_ = None  # p
        self.x_weights_ = None  # w
        self.x_scores_ = None  # t
        self.y_scores_ = None  # u
        self.x_scores_ = None  # t_ortho
        self.x_loadings_ = None  # p_ortho
        self.x_weights_ = None  # w_ortho
        self.coef_ = None  # B_pls
        self.n_components = n_components
        self.scale_ = scale
        self.x_mean_ = None
        self.y_mean_ = None
        self.x_std_ = None
        self.y_std_ = None
        self.sum_sq_X_ = None
        self.sum_sq_Y_ = None

    @staticmethod
    def _center_scale_xy(X, Y, scale=True):
        """
        Center X, Y and scale if scale parameter is True
        :param X:
        :param Y:
        :param scale:
        :return: X, Y, x_mean, y_mean, x_std, y_std
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
        w_ortho = np.zeros((np.asarray(X).shape[1], self.n_components))
        p_ortho = np.zeros((np.asarray(X).shape[1], self.n_components))
        t_ortho = np.zeros((len(X), self.n_components))
        X = check_array(X)
        Y = check_array(Y)

        if Y.shape != (X.shape[0], 1):
            raise ValueError('This OPLS implementation does not support multiple Y. '
                             'Y must be a (n_samples, 1) array-like.')

        X_res, Y_res, self.x_mean_, self.y_mean_, self.x_std_, self.y_std_ = self._center_scale_xy(X, Y, self.scale_)

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
            # X_res = X_res - np.asmatrix(t_ortho[:, i]).T @ np.asmatrix(p_ortho[:, i])
            X_res = X_res - t_ortho[:, i][:, np.newaxis] @ p_ortho[:, i][np.newaxis, :]

        # PLS on full data
        # find PLS component
        w = ((Y_res.T @ X_res) / (Y_res.T @ Y_res)).T
        w = w / np.linalg.norm(w)
        t = (X_res @ w) / (w.T @ w)
        c = (((t.T @ Y_res) / (t.T @ t)).T).item()  # this only works with single-column y
        u = (Y_res * c) / (c ** 2)
        p = ((t.T @ X_res) / (t.T @ t)).T
        # b coef
        b_l = ((1.0 / (t.T @ t)) * (u.T @ t)).item()

        self.b_ = b_l  # not sure what "b" is really...
        # self.y_loadings_= ??
        self.y_weights_ = c
        self.y_scores_ = u
        self.x_scores_ = t_ortho
        self.x_loadings_ = p_ortho
        self.x_weights_ = w_ortho
        self.sum_sq_X_ = SS_X
        self.sum_sq_Y_ = SS_Y

        # Original space
        W_star = w * (1.0 / (p.T @ w))
        B_pls = (W_star * b_l * c)
        self.coef_ = B_pls.reshape((B_pls.size, 1))

        return self

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X, y)

    def get_params(self, deep=True):
        return {'n_components': self.n_components, 'scale': self.scale_}

    def predict(self, X):
        """
        Apply the dimension reduction learned on the train data
        :param X: array-like, shape = [n_samples, n_features]. Training vectors.
        :return:
        """
        m = np.mean(X, axis=0)
        z = np.asarray(X) - m[np.newaxis, :]

        # filter out orthogonal components of X
        for f in range(0, self.n_components):
            z = (z - (z @ self.x_weights_[:, f][:, np.newaxis] / (self.x_weights_[:, f].T @ self.x_weights_[:, f])) @ self.x_loadings_[:, f][np.newaxis, :])
        return np.dot(z, self.coef_) + self.y_mean_

    def transform(self, X, Y=None):
        """
        Apply the dimension reduction learned via fit()
        :param X: array-like, shape = [n_samples, n_features]
        :param Y: array-like, shape = [n_samples, 1] (for now)
        :param copy:
        :return: (x_scores, y_scores)
        """
        check_is_fitted(self, 'x_mean_')
        X = check_array(X, dtype=FLOAT_DTYPES)
        x_scores = np.dot((X - self.x_mean_) / self.x_std_, self.x_weights_)
        if Y is not None:
            Y = check_array(Y)
            if Y.ndim == 1:
                Y = Y.reshape(-1, 1)
            y_scores = np.dot((Y - self.y_mean_) / self.y_std_, self.y_weights_)
            return x_scores, y_scores
        return x_scores

