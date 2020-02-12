# Copyright (c) 2019 Wright State University
# Author: Daniel Foose <foose.3@wright.edu>
# License: MIT

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_consistent_length


def _center_scale_xy(X, Y, scale=True):
    """ Center X, Y and scale if the scale parameter==True

    Returns
    -------
        X, Y, x_mean, y_mean, x_std, y_std
    """
    # center
    x_mean = X.mean(axis=0)
    X -= x_mean
    y_mean = Y.mean(axis=0)
    Y -= y_mean
    # scale
    if scale:
        x_std = X.std(axis=0, ddof=1)
        x_std[x_std == 0.0] = 1.0
        X /= x_std
        y_std = Y.std(axis=0, ddof=1)
        y_std[y_std == 0.0] = 1.0
        Y /= y_std
    else:
        x_std = np.ones(X.shape[1])
        y_std = np.ones(Y.shape[1])
    return X, Y, x_mean, y_mean, x_std, y_std


class OPLS(BaseEstimator, TransformerMixin):
    """Orthogonal Projection to Latent Structures (O-PLS)

    This class implements the O-PLS algorithm for one (and only one) response as described by [Trygg 2002].
    This is equivalent to the implementation of the libPLS MATLAB library (http://libpls.net/)

    Parameters
    ----------
    n_components: int, number of orthogonal components to filter. (default 5).

    scale: boolean, scale data? (default True)

    Attributes
    ----------
    W_ortho_ : weights orthogonal to y

    P_ortho_ : loadings orthogonal to y

    T_ortho_ : scores orthogonal to y

    x_mean_ : mean of the X provided to fit()
    y_mean_ : mean of the Y provided to fit()
    x_std_ : std deviation of the X provided to fit()
    y_std_ : std deviation of the Y provided to fit()

    References
    ----------
    Johan Trygg and Svante Wold. Orthogonal projections to latent structures (O-PLS).
    J. Chemometrics 2002; 16: 119-128. DOI: 10.1002/cem.695
    """
    def __init__(self, n_components=5, scale=True):
        self.n_components = n_components
        self.scale = scale

        self.W_ortho_ = None
        self.P_ortho_ = None
        self.T_ortho_ = None

        self.x_mean_ = None
        self.y_mean_ = None
        self.x_std_ = None
        self.y_std_ = None

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

        # copy since this will contains the residuals (deflated) matrices
        check_consistent_length(X, Y)
        X = check_array(X, dtype=np.float64, copy=True, ensure_min_samples=2)
        Y = check_array(Y, dtype=np.float64, copy=True, ensure_2d=False)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        X, Y, self.x_mean_, self.y_mean_, self.x_std_, self.y_std_ = _center_scale_xy(X, Y, self.scale)

        Z = X.copy()
        w = np.dot(X.T, Y)  # calculate weight vector
        w /= np.linalg.norm(w)  # normalize weight vector

        W_ortho = []
        T_ortho = []
        P_ortho = []

        for i in range(self.n_components):
            t = np.dot(Z, w)  # scores vector
            p = np.dot(Z.T, t) / np.dot(t.T, t).item()  # loadings of X
            w_ortho = p - np.dot(w.T, p).item() / np.dot(w.T, w).item() * w  # orthogonal weight
            w_ortho = w_ortho / np.linalg.norm(w_ortho)  # normalize orthogonal weight
            t_ortho = np.dot(Z, w_ortho)  # orthogonal components
            p_ortho = np.dot(Z.T, t_ortho) / np.dot(t_ortho.T, t_ortho).item()
            Z -= np.dot(t_ortho, p_ortho.T)
            W_ortho.append(w_ortho)
            T_ortho.append(t_ortho)
            P_ortho.append(p_ortho)

        self.W_ortho_ = np.hstack(W_ortho)
        self.T_ortho_ = np.hstack(T_ortho)
        self.P_ortho_ = np.hstack(P_ortho)

        return self

    def transform(self, X):
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
        Z = check_array(X, copy=True)

        Z -= self.x_mean_
        if self.scale:
            Z /= self.x_std_

        # filter out orthogonal components of X
        for i in range(self.n_components):
            t = np.dot(Z, self.W_ortho_[:, i]).reshape(-1, 1)
            Z -= np.dot(t, self.P_ortho_[:, i].T.reshape(1, -1))

        return Z

    def fit_transform(self, X, y=None, **fit_params):
        """ Learn and apply the filtering on the training data and get the filtered X

        Parameters
        ----------
        X : array-like, shape=[n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of predictors.

        y : array-like, shape = [n_samples, 1]
            Target vector, where n_samples is the number of samples.
            This O-PLS implementation only supports a single response (target) variable.
            Y=None will raise ValueError from fit().

        Returns
        -------
        X_filtered
        """
        return self.fit(X, y).transform(X)

    def score(self, X):
        """ Return the coefficient of determination R^2X of the transformation.
        Parameters
        ----------
          X : array-like of shape (n_samples, n_features)
              Test samples. For some estimators this may be a
              precomputed kernel matrix or a list of generic objects instead,
              shape = (n_samples, n_samples_fitted),
              where n_samples_fitted is the number of
              samples used in the fitting for the estimator.
          Returns
          -------
          score : float
              The amount of variation in X explained by the transformed X. A lower number indicates more orthogonal
              variation has been removed.
        """
        X = check_array(X)
        Z = self.transform(X)
        return np.sum(np.square(Z)) / np.sum(np.square(X - self.x_mean_))  # Z is already properly centered
