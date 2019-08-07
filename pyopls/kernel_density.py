import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import GridSearchCV, KFold, LeaveOneOut
from sklearn.neighbors import KernelDensity


class OPLSKernelDensity:
    @staticmethod
    def _estimate_bandwidth(vals, grid_search_num, cv, n_jobs, verbose, pre_dispatch):
        grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                            {'bandwidth': 10 ** np.linspace(-1, 1, grid_search_num)},
                            cv=cv, n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch, iid=False)
        grid.fit(vals.reshape(-1, 1))
        return grid.best_params_['bandwidth']

    @staticmethod
    def _kde(x, vals, bw):
        kd = KernelDensity(kernel='gaussian', bandwidth=bw).fit(vals.reshape(-1, 1))
        return kd.score_samples(x.reshape(-1, 1))

    @staticmethod
    def _estimate_kde_abscissa(vals, num):
        return np.linspace(vals.min(), vals.max(), num)

    def get_kdes(self, opls_cv, num=None, bandwidth=None, k=5, grid_search_num=100,
                 n_jobs=None, verbose=0, pre_dispatch='2*n_jobs'):
        # Get a kernel-density estimate for permutation test results
        num = num or 2 * opls_cv.n_permutations

        def _named_kde(key, x, vals, bw):
            return key, self._kde(x, vals, bw)

        def _named_abscissa(key, vals, n):
            return key, self._estimate_kde_abscissa(vals, n)

        if k == -1:
            cv = LeaveOneOut()
        else:
            cv = KFold(k)
        loading_bandwidths = [
            self._estimate_bandwidth(vals, grid_search_num, cv, n_jobs, verbose, pre_dispatch)
            for vals in np.hsplit(opls_cv.permutation_loadings_, opls_cv.permutation_loadings_.shape[1])
        ] if bandwidth is None else [bandwidth for _ in range(opls_cv.permutation_loadings_.shape[1])]

        loading_abscissae = Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)(
            delayed(self._estimate_kde_abscissa)(vals, num)
            for vals in np.hsplit(opls_cv.permutation_loadings_, opls_cv.permutation_loadings_.shape[1])
        )
        loading_kdes = Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)(
            delayed(self._kde)(x, vals, bw)
            for x, vals, bw in zip(loading_abscissae,
                                   np.hsplit(opls_cv.permutation_loadings_, opls_cv.permutation_loadings_.shape[1]),
                                   loading_bandwidths)
        )
        results = {
            'loadings': {
                'x': np.column_stack(loading_abscissae),
                'kde': np.column_stack(loading_kdes),
                'h': np.hstack(loading_bandwidths)
            }
        }
        metrics = {
            'q_squared': opls_cv.permutation_q_squared_,
            'r_squared_Y': opls_cv.permutation_r_squared_Y_,
            'discriminator_q_squared': opls_cv.permutation_discriminator_q_squared_,
            'accuracy': opls_cv.permutation_accuracy_,
            'roc_auc': opls_cv.permutation_roc_auc_
        }
        metrics = {key: value for key, value in metrics.items() if value is not None}
        metric_bandwidths = {
            key: self._estimate_bandwidth(value, grid_search_num, cv, n_jobs, verbose, pre_dispatch)
            for key, value in metrics.items()
        } if bandwidth is None else {key: bandwidth for key in metrics.keys()}
        metric_abscissae = {
            res[0]: res[1] for res in Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)(
                delayed(_named_abscissa)(key, value, num) for key, value in metrics.items())
        }
        metric_kdes = {
            res[0]: res[1] for res in Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)(
                delayed(_named_kde)(key, metric_abscissae[key], value, metric_bandwidths[key])
                for key, value in metrics.items()
            )
        }

        for key in metrics.keys():
            results[key] = {
                'x': metric_abscissae[key],
                'kde': metric_kdes[key],
                'h': metric_bandwidths[key]
            }

        return results
