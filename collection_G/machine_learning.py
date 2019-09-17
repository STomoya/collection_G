"""
Functions around machine learning
"""

# scikit-learn must be installed
try:
    import sklearn
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder
    import numpy as np
except:
    sklearn = None

# Exceptions
class MachineLearningException(Exception):
    """base exception class for machine learning"""
def _raise_no_sklearn_exception():
    raise MachineLearningException("No \'sklearn\' installed in your machine.")

def split_data(X, y, validation=True, random_state=None, val_random_state=None, test_size=0.2, val_size=0.2):
    """
    splits data, using 'train_test_split'

    arguments
        X : array
            The data.
        y : array
            The true labels.
        validation : bool (default : True)
            If true, the training data will be split into
            training data, and validation data
        random_state : any (default : None)
            The seed of the random state used in 'train_test_split'
        val_random_state : any (default : None)
            The seed of the random state used in 'train_test_split',
            used when splitting the generated training data.
        test_size : float between 0 to 1 (default : 0.2)
            The size of the test data.
        val_size : float between 0 to 1 (default : 0.2)
            The size of the test data,
            used when splitting the generated training data.

    returns
        X_train : array
            Training data.
        X_val : array
            Validation data.
            Only returned when 'validation' is true.
        X_test : array
            Test data.
        y_train : array
            Labels for X_train.
        y_val : array
            Labels for X_val.
            Only returned when 'validation' is true.
        y_test : array
            Labels for y_test.
    """
    if not sklearn:
        _raise_no_sklearn_exception()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, test_size=0.2)
    if validation:
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=val_random_state, test_size=val_size)
        return X_train, X_val, X_test, y_train, y_val, y_test
    else:
        return X_train, X_test, y_train, y_test

def encode_target(target, target_name=True):
    """
    Encode target to one-hot vectors

    arguments
        target : array
            The true labels of the data.
        target_name : bool (default : True)
            If true, the function will return the names of the unique labels.

    returns
        onehot_targets : array
            The labels, encoded to one-hot vectors
    """
    if not sklearn:
        _raise_no_sklearn_exception()
    try:
        reshaped = target.reshape(-1, 1)
    except AttributeError:
        np_target = np.array(target)
        reshaped = np_target.reshape(-1, 1)
    
    encoder = OneHotEncoder()
    onehot_targets = encoder.fit_transform(reshaped)
    if not target_name:
        return onehot_targets
    else:
        target_names = encoder.get_feature_names()
        return onehot_targets, target_names

