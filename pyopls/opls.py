import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, MultiOutputMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, FLOAT_DTYPES


class OPLS(BaseEstimator, TransformerMixin, RegressorMixin, MultiOutputMixin):
    def __init__(self, n_components=2, scale=True):
        self.b = None
        self.b_l = None
        self.y_weights_ = None  # c
        self.x_loadings_ = None  # p
        self.x_weights_ = None  # w
        self.x_scores_ = None  # t
        self.y_scores_ = None  # u
        self.orthogonal_x_scores_ = None  # t_ortho
        self.orthogonal_x_loadings_ = None  # p_ortho
        self.orthogonal_x_weights_ = None  # w_ortho
        self.coef_ = None  # B_pls
        self.n_components = n_components
        self.scale_ = scale
        self.Y_pred_ = None
        self.R2_X_ = None
        self.R2_Y_ = None
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

        X_res, Y_res, self.x_mean_, self.y_mean_, self.x_std_, self.y_std_ = self._center_scale_xy(X, Y, self.scale)

        X_res = np.array(X - np.mean(X, 0))  # mean-center X
        Y_res = np.array(Y - np.mean(Y, 0)).reshape((len(X), -1))  # mean-center Y
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

        self.b = b_l
        self.b_l = b_l
        self.y_weights_ = c
        self.x_loadings_ = p
        self.x_weights_ = w
        self.x_scores_ = t
        self.y_scores_ = u
        self.orthogonal_x_scores_ = t_ortho
        self.orthogonal_x_loadings_ = p_ortho
        self.orthogonal_x_weights_ = w_ortho
        self.sum_sq_X_ = SS_X
        self.sum_sq_Y_ = SS_Y

        # Original space
        W_star = w * (1.0 / (p.T @ w))
        B_pls = (W_star * b_l * c)
        self.coef_ = B_pls.reshape((B_pls.size, 1))

        m = np.mean(X, axis=0)
        X_res = np.asarray(X) - m[np.newaxis, :]
        z = X_res

        # filter out OPLS components
        for f in range(0, self.n_components):
            z = (z - (z @ w_ortho[:, f][:, np.newaxis] / (w_ortho[:, f].T @ w_ortho[:, f])) @ p_ortho[:, f][np.newaxis, :])

        # predict
        #  self.Y_pred = (z @ self.B_pls) + np.mean(Y, axis=0)
        # self.R2_X = float((t.T @ t) * (p.T @ p) / SS_X)
        # Use score() inherited from sklearn base
        # self.R2_Y = float((t.T @ t) * (b_l ** 2) * (c ** 2) / SS_Y)

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y).transform(X, y)

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
            z = (z - (z @ self.orthogonal_x_weights_[:, f][:, np.newaxis] / (self.orthogonal_x_weights_[:, f].T @ self.orthogonal_x_weights_[:, f])) @ self.orthogonal_x_loadings_[:, f][np.newaxis, :])
        return (z @ self.B_pls) + self.y_mean_

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
        Xr = (X - self.x_mean_) / self.x_std_
        x_scores = Xr @ self.orthogonal_x_weights_
        if Y is not None:
            if Y.ndim == 1:
                Y = Y.reshape(-1, 1)
            Yr = (Y - self.y_mean_) / self.y_std_
            y_scores = Yr @ self.y_weights_
            return x_scores, y_scores
        return x_scores

