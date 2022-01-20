"""Convert and preprocess data for ML complete intersection investigations."""

import sqlite3
import pandas as pd
import numpy as np

from sklearn.utils import shuffle

# pylint: disable=invalid-name


def _string_to_list(s, n_terms):
    """Convert string to list
    Args:
        s:          string of comma separated floats (e.g., "1.0,3.5,2.8")
                    (i-th float in list denoted x[i])
        n_terms:    number of terms to be included in final list

    Returns:
        list of floats of len n_terms [x[i] for i in range(n_terms)]"""
    return np.array([np.float32(x) for x in s.split(',')[:n_terms]],
                    dtype=np.float32)


def load_data(n_terms):
    """Load and preprocess complete intersections data
    Args:
        n_terms:    positive integer indicating the number of terms to be used

    Returns:
        X, y two numpy arrays
        X:  numpy array containing the preprocessed Hilbert series
            [hs[i]/hs[i+1]|i=0,...,n_terms]
        y:  integer indicating if HS comes from a complete intersection (1) or
            non-complete intersection (0)"""
    print("Loading data...")

    # load and pre-process data
    with sqlite3.connect('../../Data/ci_big.db') as db:
        ci = pd.read_sql_query('SELECT hs,ci FROM ci;', db)

    ci['hs'] = ci['hs'].apply(lambda s: _string_to_list(s, n_terms))

    X = np.stack(ci['hs'].to_numpy(), axis=0)
    y = ci['ci'].to_numpy()

    print("\nFinished loading data...")

    # shuffle data to ensure that ci and non-ci samples are randomly distributed
    return shuffle(X, y)
