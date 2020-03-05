def test_opls():
    import numpy as np
    from pyopls import OPLS
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.metrics import r2_score
    from sklearn.model_selection import cross_val_predict, LeaveOneOut

    # paths relative to repo
    spectra = np.load('pyopls/tests/features.npy')
    target = np.load('pyopls/tests/target.npy')

    score = -1
    n_components = 0
    for n_components in range(1, spectra.shape[1]):
        opls = OPLS(n_components=n_components)
        Z = opls.fit(spectra, target).transform(spectra)
        y_pred = cross_val_predict(PLSRegression(n_components=1), Z, target, cv=LeaveOneOut())
        score_i = r2_score(target, y_pred)
        if score_i < score:
            n_components -= 1
            break
        score = score_i

    opls = OPLS(n_components=n_components)
    opls.fit(spectra, target)
    assert opls.n_components == n_components
    assert opls.P_ortho_.shape == (spectra.shape[1], n_components)
    assert opls.T_ortho_.shape == (spectra.shape[0], n_components)
    assert opls.W_ortho_.shape == (spectra.shape[1], n_components)
    assert opls.x_mean_.shape == (spectra.shape[1],)
    assert opls.x_std_.shape == (spectra.shape[1],)
    assert opls.y_mean_.shape == (1,)
    assert opls.y_std_.shape == (1,)

    Z = opls.transform(spectra)
    assert Z.shape == spectra.shape

    pls = PLSRegression(n_components=1)
    uncorrected_r2 = r2_score(target, pls.fit(spectra, target).predict(spectra))
    corrected_r2 = r2_score(target, pls.fit(Z, target).predict(Z))
    uncorrected_q2 = r2_score(target, cross_val_predict(pls, spectra, target, cv=LeaveOneOut()))
    corrected_q2 = r2_score(target, cross_val_predict(pls, Z, target, cv=LeaveOneOut()))

    assert uncorrected_r2 < corrected_r2
    assert uncorrected_q2 < corrected_q2
