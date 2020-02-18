import pytest
from sklearn.base import clone

def test_clone():
    # Tests that clone creates a correct deep copy.
    # We create an estimator, make a copy of its original state
    # (which, in this case, is the current state of the estimator),
    # and check that the obtained copy is a correct deep copy.

    from pyopls import OPLS
    
    estimator = OPLS(n_components=2)
    new_estimator = clone(estimator)
    assert estimator is not new_estimator
    assert estimator.get_params() == new_estimator.get_params()

    estimator = OPLS(n_components=5)
    new_estimator = clone(estimator)
    assert estimator is not new_estimator
