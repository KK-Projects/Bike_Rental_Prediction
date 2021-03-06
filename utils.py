import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.cross_validation import StratifiedKFold


def df_train_test_split(data_X, data_Y, test_size):
    """
    Function that gives the same result as train_test_split but as dataframe
    :param data_X:
    :param data_Y:
    :param test_size:
    :return:
    """
    features = data_X.columns
    output = data_Y.columns

    X_train, X_test, Y_train, Y_test = train_test_split(data_X, data_Y, test_size=test_size)

    df_X_train = pd.DataFrame(X_train)
    df_X_test = pd.DataFrame(X_test)
    df_Y_train = pd.DataFrame(Y_train)
    df_Y_test = pd.DataFrame(Y_test)

    df_X_train.columns = features
    df_X_test.columns = features
    df_Y_train.columns = output
    df_Y_test.columns = output

    return df_X_train, df_X_test, df_Y_train, df_Y_test


def subdivise_data(data_X, data_Y, test_size, number_sets):
    """
    This function takes the data of some inputs data_X and variable that you want to predict Y, and returns
    a dictionnary number of sets of train and test data according to the test size ( = data_test size / data_X)
    :param data_X: Dataframe
    :param data_Y: Dataframe
    :param test_size: float between 0 and 1
    :param number_sets:
    :return: Dict
    """
    dict_train = {}
    for i in range(number_sets):

        df_X_train, df_X_test, df_Y_train, df_Y_test = df_train_test_split(data_X, data_Y, test_size)

        set = 'set_{}'.format(i)
        dict_train[set] = {}
        dict_train[set]['X_train'] = df_X_train
        dict_train[set]['X_test'] = df_X_test
        dict_train[set]['Y_train'] = df_Y_train
        dict_train[set]['Y_test'] = df_Y_test

    return dict_train


def cat_var_to_dummies(categorical_input_X):
    """
    :param categorical_input_X: DataFrame of categorical variables
    :return:
    """

    df = pd.DataFrame()
    for col in categorical_input_X.columns:
        col_dummies = pd.get_dummies(categorical_input_X[col], prefix=col)
        df = pd.concat([df, col_dummies], axis=1)

    return df


def get_my_input(_input_X, cat_feat, non_cat_feat, bucks={}, buckets=False, drop_feat=False):
    """

    :param _input_X:
    :param cat_feat:
    :param non_cat_feat:
    :param bucks_feat:
    :param bucks:
    :param buckets:
    :return:
    """
    categorical_input_X = _input_X[cat_feat]
    non_cat_input_X = _input_X[non_cat_feat]

    if buckets:
        bucks_feat = bucks.keys()
        for b in bucks_feat:
            if b in cat_feat:
                out = pd.cut(categorical_input_X[b], bins=bucks[b])
            elif b in non_cat_feat:
                out = pd.cut(non_cat_input_X[b], bins=bucks[b])
            out = out.astype(str)
            categorical_input_X[b + '_bucket'] = out
            if drop_feat:
                categorical_input_X = categorical_input_X.drop(b, 1)

    cat_input_X = cat_var_to_dummies(categorical_input_X)
    input_X = pd.concat([cat_input_X, non_cat_input_X], axis=1)

    return input_X


def scale_non_cat_feat(input_train, input_test, non_cat_feat):
    """

    :param _input_X:
    :param cat_feat:
    :param non_cat_feat:
    :return:
    """
    scaler = preprocessing.StandardScaler()
    cat_feat = input_train.columns.drop(non_cat_feat)
    categorical_input_train = input_train[cat_feat]
    non_cat_input_train = input_train[non_cat_feat]
    cat_input_train = pd.DataFrame(scaler.fit_transform(categorical_input_train))
    cat_input_train.columns = cat_feat

    _input_train = pd.concat([cat_input_train, non_cat_input_train], axis=1)

    categorical_input_test = input_test[cat_feat]
    non_cat_input_test = input_test[non_cat_feat]
    cat_input_test= pd.DataFrame(scaler.transform(categorical_input_test))
    cat_input_test.columns = cat_feat

    _input_test= pd.concat([cat_input_test, non_cat_input_test], axis=1)

    return _input_train, _input_test


def cross_validate(_input_X, _input_Y, classifier, nb_folds=10, non_cat_feat=['temp', 'atemp', 'hum', 'windspeed']):
    """ Perform a cross-validation and returns the predictions.
        Use a scaler to scale the features to mean 0, standard deviation 1.

        Parameters:
        -----------
        _input_X: (n_samples, n_features) np.array
        _input_Y: (n_samples, ) np.array
        classifier:  sklearn classifier object
            Classifier instance; must have the following methods:
            - fit(X, y) to train the classifier on the data X, y
            - predict(X) to apply the trained classifier to the data X and return probability estimates
        nb_folds: snumber of folds for the cross validation

        Return:
        -------
        pred: (n_samples, ) np.array
            Vectors of predictions
        """
    cv_folds = StratifiedKFold(np.array(_input_Y).reshape(_input_Y.shape[0]), nb_folds, shuffle=True)
    pred = np.zeros(_input_Y.shape)  # vector of 0 in which to store the predictions
    for tr, te in cv_folds:
        # Restrict data to train/test folds
        Xtr = np.array(_input_X)[tr, :]
        ytr = np.array(_input_Y)[tr]
        Xte = np.array(_input_X)[te, :]
        yte = np.array(_input_Y)[te]

        # Scale data
        scaler = preprocessing.StandardScaler()  # create scaler
        Xtr = scaler.fit_transform(Xtr)  # fit the scaler to the training data and transform training data
        Xte = scaler.transform(Xte)  # transform test data

        # Fit classifier
        classifier.fit(Xtr, ytr)

        # Predictions
        pred[te] = classifier.predict(Xte).reshape(yte.size,1)
    return pred


def cross_validate_with_grid(_input_X, _input_Y, classifier, nb_folds):
    """ Perform a cross-validation and returns the predictions.
    Use a scaler to scale the features to mean 0, standard deviation 1.

    Parameters:
    -----------
    design_matrix: (n_samples, n_features) np.array
        Design matrix for the experiment.
    labels: (n_samples, ) np.array
        Vector of labels.
    classifier:  sklearn classifier object
        Classifier instance; must have the following methods:
        - fit(X, y) to train the classifier on the data X, y
        - predict_proba(X) to apply the trained classifier to the data X and return probability estimates
    cv_folds: sklearn cross-validation object
        Cross-validation iterator.

    Return:
    -------
    pred: (n_samples, ) np.array
        Vectors of predictions (same order as labels).
    """
    cv_folds = StratifiedKFold(_input_Y, nb_folds, shuffle=True)
    pred = np.zeros(_input_Y.shape)  # vector of 0 in which to store the predictions
    for tr, te in cv_folds:
        # Restrict data to train/test folds
        Xtr = _input_X[tr, :]
        ytr = _input_Y[tr]
        Xte = _input_X[te, :]
        # print Xtr.shape, ytr.shape, Xte.shape

        # Scale data
        scaler = preprocessing.StandardScaler()  # create scaler
        Xtr = scaler.fit_transform(Xtr)  # fit the scaler to the training data and transform training data
        Xte = scaler.transform(Xte)  # transform test data

        # Fit classifier
        classifier.fit(Xtr, ytr)

        # Predict probabilities (of belonging to +1 class) on test data
        pred[te] = classifier.predict(Xte)  # two-dimensional array
    return pred
