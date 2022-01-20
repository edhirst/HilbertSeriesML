"""Scikit-learn pipelines for classification questions on Hilbert series."""

import scipy.stats
import numpy as np

from sklearn.base import TransformerMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef
from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle

# pylint: disable=invalid-name


class LogNormalise(TransformerMixin):
    """LogNormalise sklearn transformer"""

    def fit(self, X, y=None):
        """Do nothing"""
        # pylint: disable=unused-argument
        return self

    def transform(self, X):
        """Take component-wise (extended) log of X.

        Extend log(x): -1 if x <= 0 else usual log(x)

        Args:
            X: numpy array of numpy arrays of numerical values

        Returns:
            numpy array of numpy arrays of float32 obtained by taking
            component-wise (extended) log of entries of X.
        """
        # pylint: disable=no-self-use
        return np.array([np.log(x.astype("float32"),
                                out=-np.ones_like(x).astype("float32"),
                                where=(x > 0)) for x in X])


def _pipeline_setup(log_normalise, use_pca=True):
    """Return sklearn pipeline: (optional) log normaliser + PCA + random forest.
    (pipeline is processed in this order)

    Args:
        log_normalise: Boolean wheather log normaliser should be included

    Returns:
        sklearn pipeline with transformers:
            log_normaliser: if log_normalise is true
            PCA: with n_components=0.99999
                 (we would like to set 1.0, but sklearn doesn't allow it)
            RandomForestClassifier: standard parameter
    """
    pipe_list = []

    if log_normalise:
        pipe_list.append(('log_normalise', LogNormalise()))

    if use_pca:
        # we wish we could write '1.0', but sklearn doesn't let us
        pipe_list.append(('pca', PCA(n_components=0.99999)))

    pipe_list.append(('random_forest_classifier', RandomForestClassifier()))

    return Pipeline(pipe_list)


def run_pipeline(X, y, test_size_, log_normalise=True, use_pca=True):
    """Prints statistical scores of sklearn pipeline (run once).

    Sklearn pipeline: (optional) LogNormalise + PCA + RandomForestClassifier

    Prints [accuracy, confusion_matrix, matthews_corrcoef].

    Args:
        X: numpy array of numpy arrays of numerical values
        y: numpy array of integer labels
        test_size_: value between 0.0 and 1.0 determining the test size
        log_normalise: boolean (default=True)
    """
    hs_pipeline = _pipeline_setup(log_normalise, use_pca)

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size_)

    clf = hs_pipeline.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"\nrandom forest accuracy = \
    {round(accuracy_score(y_pred, y_test)*100, 2)}%")
    print(f"\nconfusion_matrix = \n{confusion_matrix(y_test, y_pred)}")
    print(f"\nmcc={matthews_corrcoef(y_test, y_pred)}")


def run_pipeline_cross_val(X,
                           y,
                           cv_,
                           invert_folds=True,
                           log_normalise=True,
                           use_pca=True):
    """Prints statistical scores of k-fold cross-validation of sklearn pipeline.

    Sklearn pipeline: (optional) LogNormalise + PCA + RandomForestClassifier

    Prints means and standard deviations of
    [accuracy, matthews_corrcoef, confusion_matrix].

    Args:
        X: numpy array of numpy arrays of numerical values
        y: numpy array of integer labels
        cv_: number of folds for cross-validation
        invert_folds: boolean whether current chunck of k-fold corss-validation
                      used for training (invert_folds=True) or the remaining
                      chuncks (invert_folds=False).
        log_normalise: boolean (default=True)
    """
    #pylint: disable=too-many-arguments
    # pylint: disable=too-many-locals
    hs_pipeline = _pipeline_setup(log_normalise, use_pca)
    X_, y_ = shuffle(X, y)

    kf = KFold(n_splits=cv_)
    accuracy_scores = []
    mcc_scores = []
    cm_scores = []
    for train, test in kf.split(X_):
        if invert_folds:
            test, train = train, test
        clf = hs_pipeline.fit(X_[train], y_[train])
        y_pred = clf.predict(X_[test])
        accuracy_scores.append(accuracy_score(y_pred, y_[test]))
        mcc_scores.append(matthews_corrcoef(y_[test], y_pred))
        cm_scores.append(confusion_matrix(y_[test], y_pred))

    accuracy_std_deviation = np.std(accuracy_scores)
    print("accuracy:")
    print(f"mean: {np.mean(accuracy_scores)}")
    print(f"standard deviation: {accuracy_std_deviation}")
    # compute confidence interval
    cfi = scipy.stats.t.interval(0.95,
                                 len(accuracy_scores)-1,
                                 loc=np.mean(accuracy_scores),
                                 scale=scipy.stats.sem(accuracy_scores))
    print(f"confidence interval of 95%: +/-{(cfi[1]-cfi[0])/2}\n")

    print("mcc:")
    print(f"Mean: {np.mean(mcc_scores)}")
    print(f"Standard deviation: {np.std(mcc_scores)}\n")

    print("cm:")
    print(f"Mean:\n{np.mean(cm_scores, axis=0)}\n")
    print(f"Standard deviation:\n{np.std(cm_scores, axis=0)}\n")
